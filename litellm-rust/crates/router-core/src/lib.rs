//! Routing for LiteLLM: decide which deployment serves a request, and drive retries, fallbacks
//! and cooldowns around the attempts.
//!
//! This crate defines the boundary `litellm.Router` is moving behind. Today the Python router in
//! `litellm/router.py` implements all of it; this crate is where the Rust core will implement the
//! same interface. Until then it holds this overview and nothing else, so the interface is written
//! down in code before either side changes.
//!
//! # The public surface
//! Code outside the router uses `litellm.Router` through two kinds of members and nothing else:
//! - The endpoint methods (`acompletion`, `aembedding`, `aresponses`, `aanthropic_messages`, the
//!   files, batches, assistants and vector store wrappers, 37 in all). Each is a thin wrapper over
//!   `call(op, request)`.
//! - Eight entry points: `registry()`, `settings()`, `health()`, `select()`, `apply()`,
//!   `observe()`, `register_plugin()` and `retire()`.
//!
//! Every other member that outside code uses today has a named replacement among those. The old
//! names stay available for about a month through a `DeprecatedRouterMembers` mixin, which
//! implements them over the entry points and warns on use. A CI check
//! (`tests/code_coverage_tests/router_surface_check.py`) keeps an allowlist of the members outside
//! code touches; the list may only shrink.
//!
//! # Structure
//! - `Registry`: the configured deployments, indexed by id, model group, alias, wildcard pattern
//!   and team. Immutable and versioned. Seven queries: `deployment(id)`, `deployments(filter)`,
//!   `names(filter)`, `resolve(name, scope)`, `group(name)`, `access_groups()` and
//!   `credentials(id)`. "Registry" is the router's view of what is configured; per-model pricing
//!   and capabilities are the model catalog (`litellm-model-catalog`), which the router reads
//!   through its dynamic config.
//! - `RouterConfig` / `RouterDynamicConfig`: everything the router needs to know, passed in. The
//!   router never reads process globals. Static config (model list, strategy, retry, timeout and
//!   cooldown policy, Redis settings, plugins) is fixed for the router's life. Dynamic config
//!   (pricing, model aliases, fallbacks, retry counts, callbacks) is a versioned snapshot read
//!   once per call.
//! - `Settings`: one typed, frozen view of the configuration a caller may read.
//! - `Health`: cooldowns, blocked deployments and usage, kept in shared cache keys so Python and
//!   Rust workers see the same state. `report(id, outcome)` records outcomes the router did not
//!   see itself (a health check, an interceptor that marks a deployment failed).
//! - `select(model, request)`: resolve a name, filter eligible deployments, pick one with the
//!   configured strategy, without calling it.
//! - `apply(changes)`: one `ChangeSet` (upserts, deletes, settings), applied atomically, producing
//!   a new registry version.
//! - `call(op, request)`: select, run the attempt, and retry or fall back until it succeeds or the
//!   policy is exhausted.
//!
//! # Hosts: the only boundary
//! The core reaches outside itself only through [`Host`], and every attempt runs through
//! `Host::invoke`: the core picks the deployment, the host runs the call, the core records the
//! outcome. There is no second, "native" attempt path in the core. Which code runs an attempt is
//! the host's business:
//! - `PythonHost` (inside a Python process) builds the request for the picked deployment and calls
//!   `litellm.<op>`. It never calls back into a Python router.
//! - `NativeHost` (the Rust server) calls the `inference-*` crates. An op or provider it cannot run
//!   fails `Host::supports` when the router is built or changed, never partway through a request.
//!
//! Besides `invoke` and `supports`, a host provides `select` for strategies and plugins not yet
//! ported, `allow_fallback` for access and budget checks, `sleep`, `observe` for routing events,
//! and `cache`, a DualCache-style store that holds all routing state under the same keys,
//! encodings and TTLs the Python router uses. The core writes cooldowns, allowed-fails counters
//! and usage itself, through that cache.
//!
//! # Plugins
//! Routing strategies, pre-call checks and sub-routers (auto, complexity, quality, adaptive)
//! register through `register_plugin` and see a `RoutingContext` (registry, settings, health and
//! the current request), never the router itself.
//!
//! # Modules
//! - [`registry`]: deployments and their indices, `deployment_id`, `ChangeSet`.
//! - [`config`] / [`settings`]: static and dynamic config, and the frozen `Settings` view.
//! - [`cache`] / [`health`]: the shared routing cache, cooldowns and usage.
//! - [`select`]: resolve, filter chain, simple-shuffle.
//! - [`call`]: retries and fallbacks.
//! - [`host`]: the `Host` boundary.

pub mod cache;
pub mod call;
pub mod config;
pub mod error;
pub mod health;
pub mod host;
pub mod registry;
pub mod select;
pub mod settings;

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use arc_swap::ArcSwap;
use parking_lot::Mutex;

pub use crate::{
    cache::{InMemoryRouterCache, RouterCache},
    call::RouterResponse,
    config::{Op, RouterConfig, RouterDynamicConfig, RoutingStrategy},
    error::{AttemptError, ConfigError, Error, FailureKind, Payload, Unsupported},
    health::{Clock, Health, Outcome, SystemClock},
    host::{Attempt, Host, Output, RouterEvent, RouterRequest, RoutingContext, SelectStage},
    registry::{ChangeSet, Deployment, DeploymentSpec, Registry, RegistryVersion},
    select::{Randomness, SeededRandomness, ThreadRandomness},
    settings::Settings,
};
use crate::{health::GroupShape, registry::DeploymentFilter, select::SelectInputs};

/// The registry and settings one call reads, swapped atomically by `apply`.
pub(crate) struct Snapshot {
    pub registry: Arc<Registry>,
    pub settings: Arc<Settings>,
}

pub(crate) struct Core {
    host: Arc<dyn Host>,
    rng: Arc<dyn Randomness>,
    health: Health,
    config: RouterConfig,
    snapshot: ArcSwap<Snapshot>,
    writer: Mutex<()>,
    retired: AtomicBool,
}

impl Core {
    /// Reject what this host cannot serve, before any request.
    fn validate(&self, registry: &Registry, settings: &Settings) -> Result<(), ConfigError> {
        let can_select = self.host.can_select();
        let strategies = std::iter::once(&settings.routing_strategy).chain(
            registry
                .routing_groups()
                .iter()
                .map(|group| &group.routing_strategy),
        );
        for strategy in strategies {
            if !strategy.is_native() && !can_select {
                return Err(Unsupported::Strategy(strategy.clone()).into());
            }
        }
        if let Some(plugin) = settings.plugins.first()
            && !can_select
        {
            return Err(Unsupported::Plugin(plugin.name.clone()).into());
        }
        for deployment in registry.deployments(&DeploymentFilter::default()) {
            for op in &self.config.served_ops {
                if !self.host.supports(op, deployment.provider()) {
                    return Err(Unsupported::Op {
                        op: op.clone(),
                        provider: deployment.provider().to_owned(),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }
}

/// Builds a [`Router`], with test seams.
pub struct RouterBuilder {
    config: RouterConfig,
    dynamic: RouterDynamicConfig,
    host: Arc<dyn Host>,
    rng: Arc<dyn Randomness>,
    clock: Arc<dyn Clock>,
}

impl RouterBuilder {
    #[must_use]
    pub fn randomness(self, rng: Arc<dyn Randomness>) -> Self {
        Self { rng, ..self }
    }

    #[must_use]
    pub fn clock(self, clock: Arc<dyn Clock>) -> Self {
        Self { clock, ..self }
    }

    pub fn build(self) -> Result<Router, ConfigError> {
        let registry = Registry::build(
            self.config.model_list.clone(),
            self.dynamic.model_alias_map.clone(),
            self.config.routing_groups.clone(),
        )?;
        let settings = Settings::new(&self.config, Arc::new(self.dynamic));
        let core = Core {
            health: Health::new(self.host.cache(), self.clock),
            host: self.host,
            rng: self.rng,
            config: self.config,
            snapshot: ArcSwap::from_pointee(Snapshot {
                registry: Arc::new(Registry::default()),
                settings: Arc::new(settings.clone()),
            }),
            writer: Mutex::new(()),
            retired: AtomicBool::new(false),
        };
        core.validate(&registry, &settings)?;
        core.snapshot.store(Arc::new(Snapshot {
            registry: Arc::new(registry),
            settings: Arc::new(settings),
        }));
        Ok(Router {
            core: Arc::new(core),
        })
    }
}

/// The router: the entry points `litellm.Router` delegates to.
#[derive(Clone)]
pub struct Router {
    core: Arc<Core>,
}

impl Router {
    pub fn new(
        config: RouterConfig,
        dynamic: RouterDynamicConfig,
        host: Arc<dyn Host>,
    ) -> Result<Self, ConfigError> {
        Self::builder(config, dynamic, host).build()
    }

    pub fn builder(
        config: RouterConfig,
        dynamic: RouterDynamicConfig,
        host: Arc<dyn Host>,
    ) -> RouterBuilder {
        RouterBuilder {
            config,
            dynamic,
            host,
            rng: Arc::new(ThreadRandomness),
            clock: Arc::new(SystemClock),
        }
    }

    fn live(&self) -> Result<(), Error> {
        if self.core.retired.load(Ordering::Acquire) {
            return Err(Error::Retired);
        }
        Ok(())
    }

    /// Select, run the attempt, and retry or fall back until it succeeds or the policy is
    /// exhausted.
    pub async fn call(&self, op: Op, request: RouterRequest) -> Result<RouterResponse, Error> {
        self.live()?;
        self.core.call(op, Arc::new(request)).await
    }

    /// Resolve, filter and pick a deployment without calling it.
    pub async fn select(&self, op: &Op, request: &RouterRequest) -> Result<Arc<Deployment>, Error> {
        self.live()?;
        let snapshot = self.core.snapshot.load_full();
        let candidates = self
            .core
            .candidates(
                &snapshot,
                op,
                request,
                &request.model,
                &SelectInputs::default(),
            )
            .await?;
        self.core.pick(&snapshot, op, request, &candidates).await
    }

    /// The deployments a pick for `request` would choose among: resolved, team- and
    /// cooldown-filtered, through the plugins and the lowest `order`.
    pub async fn healthy(
        &self,
        op: &Op,
        request: &RouterRequest,
    ) -> Result<Vec<Arc<Deployment>>, Error> {
        self.live()?;
        let snapshot = self.core.snapshot.load_full();
        let candidates = self
            .core
            .candidates(
                &snapshot,
                op,
                request,
                &request.model,
                &SelectInputs::default(),
            )
            .await?;
        Ok(candidates.healthy)
    }

    pub fn registry(&self) -> Arc<Registry> {
        self.core.snapshot.load().registry.clone()
    }

    pub fn settings(&self) -> Arc<Settings> {
        self.core.snapshot.load().settings.clone()
    }

    pub fn health(&self) -> &Health {
        &self.core.health
    }

    /// Record an outcome the router did not see itself, such as a health check. Returns whether
    /// the deployment was cooled down.
    pub async fn report(&self, id: &str, outcome: Outcome) -> Result<bool, Error> {
        let snapshot = self.core.snapshot.load_full();
        let Some(deployment) = snapshot.registry.deployment(id) else {
            return Ok(false);
        };
        match outcome {
            Outcome::Success { total_tokens } => self
                .core
                .health
                .record_success(&deployment, total_tokens)
                .await
                .map(|()| false),
            Outcome::Failure(failure) => {
                let shape = GroupShape {
                    single_deployment: snapshot
                        .registry
                        .group(&deployment.model_name)
                        .is_some_and(|group| group.len() == 1),
                    has_fallback_group: snapshot
                        .settings
                        .dynamic
                        .fallback_targets(&[deployment.model_name.as_str()])
                        .is_some(),
                };
                self.core
                    .health
                    .record_failure(&snapshot.settings, &deployment, shape, &failure)
                    .await
                    .map(|cooldown| cooldown.is_some())
            }
        }
    }

    /// Apply one change set atomically and return the new registry version.
    pub fn apply(&self, changes: ChangeSet) -> Result<RegistryVersion, ConfigError> {
        let _writer = self.core.writer.lock();
        let current = self.core.snapshot.load_full();
        let aliases = changes
            .settings
            .as_ref()
            .map(|dynamic| dynamic.model_alias_map.clone());
        let registry = current.registry.apply(&changes, aliases)?;
        let settings = match changes.settings {
            Some(dynamic) => Settings::new(&self.core.config, Arc::new(dynamic)),
            None => Settings::clone(&current.settings),
        };
        self.core.validate(&registry, &settings)?;
        let version = registry.version();
        self.core.snapshot.store(Arc::new(Snapshot {
            registry: Arc::new(registry),
            settings: Arc::new(settings),
        }));
        Ok(version)
    }

    /// Stop serving: every later call and selection fails with [`Error::Retired`].
    pub fn retire(&self) {
        self.core.retired.store(true, Ordering::Release);
    }
}
