use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use litellm_config::{RedisSettings, ResolvedRedis, SecretSnapshot};
use litellm_core_utils::settings::ProcessEnvironment;
use litellm_redis_client::{RedisClient, RedisEvents};
use litellm_redis_connect::{HostServices, SharedEnvironment};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
};

use super::events::ServiceLoggingEvents;

/// The `redis_kwargs` dict `RedisCache` hands redis-py, read the way `litellm/_redis.py` reads it.
/// A value that is not plain data, such as a `credential_provider` object, reaches the resolver by
/// its type name, so the resolver or the connector names the setting it cannot honor.
pub(crate) struct PythonRedisSettings(RedisSettings);

impl<'a, 'py> FromPyObject<'a, 'py> for PythonRedisSettings {
    type Error = PyErr;

    fn extract(kwargs: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = kwargs.py();
        let plain = PyDict::new(py);
        for (name, value) in kwargs.cast::<PyDict>()?.iter() {
            if value.is_none() {
                continue;
            }
            if is_plain(&value) {
                plain.set_item(&name, value)?;
            } else {
                plain.set_item(&name, value.get_type().name()?)?;
            }
        }
        Ok(Self(litellm_host_python::from_py_argument(plain.as_any())?))
    }
}

fn is_plain(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyString>()
        || value.is_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
        || value.is_instance_of::<PyList>()
        || value.is_instance_of::<PyTuple>()
        || value.is_instance_of::<PyDict>()
}

struct Shared {
    resolved: ResolvedRedis,
    client: RedisClient,
}

static CLIENTS: LazyLock<Mutex<Vec<Shared>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// The one client for a resolved Redis config, so every family in this process shares its pool
/// and its breaker. A config the resolver rejects fails here, at boot, rather than leaving the
/// family on memory alone.
pub(crate) fn shared_client(
    py: Python<'_>,
    kwargs: PythonRedisSettings,
    namespace: Option<String>,
) -> PyResult<RedisClient> {
    let environment: SharedEnvironment = Arc::new(ProcessEnvironment);
    let secrets = SecretSnapshot::read(kwargs.0.secret_names(), &*environment);
    let resolved = ResolvedRedis {
        namespace,
        ..kwargs.0.resolve(&secrets, &*environment).map_err(|error| {
            PyValueError::new_err(litellm_error::Report::new(&error).to_string())
        })?
    };
    let mut clients = CLIENTS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(shared) = clients.iter().find(|shared| shared.resolved == resolved) {
        return Ok(shared.client.clone());
    }
    let events: Arc<dyn RedisEvents> = Arc::new(ServiceLoggingEvents::new(py)?);
    let breaker = litellm_redis_connect::breaker_config(&*environment)
        .map_err(|error| PyValueError::new_err(litellm_error::Report::new(&error).to_string()))?;
    let host = HostServices {
        events: Some(events),
        breaker,
        http: crate::http::host_client(py, litellm_http::ClientVariant::Provider)?,
        environment,
    };
    let target = resolved.clone();
    let client = crate::execution::run_sync_value(py, async move {
        litellm_redis_connect::connect(target, host)
            .map_err(|error| PyValueError::new_err(litellm_error::Report::new(&error).to_string()))
    })?;
    clients.push(Shared {
        resolved,
        client: client.clone(),
    });
    Ok(client)
}
