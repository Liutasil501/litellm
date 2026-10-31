use std::{path::PathBuf, sync::Arc, time::Duration};

use redis::{
    ClientTlsConfig, ConnectionAddr, ConnectionInfo, IntoConnectionInfo, RedisConnectionInfo,
    TlsCertificates, TlsMode,
};

use litellm_circuit_breaker::Breaker;
use litellm_secret_value::SecretValue;

use crate::{ConnectError, Namespace, RedisEvents};

// litellm/constants.py REDIS_SOCKET_TIMEOUT, which _coerce_redis_kwargs_types sets when socket_timeout is unset
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

pub struct RedisConfig {
    pub endpoint: Endpoint,
    pub credentials: Credentials,
    pub namespace: Namespace,
    pub timeouts: Timeouts,
    pub breaker: Option<Breaker>,
    pub events: Option<Arc<dyn RedisEvents>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timeouts {
    pub response: Duration,
    pub connect: Duration,
}

impl Timeouts {
    pub fn new(socket_timeout: Duration, connect_timeout: Option<Duration>) -> Self {
        Self {
            response: socket_timeout,
            connect: connect_timeout.unwrap_or(socket_timeout),
        }
    }
}

impl Default for Timeouts {
    fn default() -> Self {
        Self::new(DEFAULT_TIMEOUT, None)
    }
}

#[derive(Clone, Debug)]
pub struct Credentials {
    pub username: Option<SecretValue>,
    pub password: Option<SecretValue>,
}

impl Credentials {
    pub const NONE: Self = Self {
        username: None,
        password: None,
    };
}

#[derive(Clone, Debug, PartialEq)]
pub enum Endpoint {
    Url { url: SecretValue, tls: TlsOptions },
    Host(HostEndpoint),
    Cluster(ClusterEndpoint),
    Sentinel(SentinelEndpoint),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HostEndpoint {
    pub host: String,
    pub port: u16,
    pub tls: Option<TlsOptions>,
    pub db: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterEndpoint {
    pub nodes: Vec<ClusterNode>,
    pub tls: Option<TlsOptions>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterNode {
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SentinelEndpoint {
    pub sentinels: Vec<ClusterNode>,
    pub service_name: String,
    pub sentinel_password: Option<SecretValue>,
    pub tls: Option<TlsOptions>,
    pub db: i64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TlsOptions {
    pub ca_certs: Option<PathBuf>,
    pub certfile: Option<PathBuf>,
    pub keyfile: Option<PathBuf>,
    pub verify_certificate: bool,
    pub check_hostname: bool,
}

impl TlsOptions {
    pub fn verified() -> Self {
        Self {
            verify_certificate: true,
            ..Self::default()
        }
    }

    pub(crate) fn mode(&self) -> TlsMode {
        match self.verify_certificate {
            true => TlsMode::Secure,
            false => TlsMode::Insecure,
        }
    }

    pub(crate) fn certificates(&self) -> Result<TlsCertificates, ConnectError> {
        let client_tls = match (&self.certfile, &self.keyfile) {
            (Some(certfile), keyfile) => Some(ClientTlsConfig {
                client_cert: read(certfile)?,
                client_key: read(keyfile.as_ref().unwrap_or(certfile))?,
            }),
            (None, _) => None,
        };
        Ok(TlsCertificates {
            client_tls,
            root_cert: self
                .ca_certs
                .as_ref()
                .map(read)
                .transpose()?
                .map(with_system_roots),
        })
    }

    pub(crate) fn apply(&self, info: ConnectionInfo) -> Result<ConnectionInfo, ConnectError> {
        let tls_info = redis::Client::build_with_tls(info, self.certificates()?)?
            .get_connection_info()
            .clone();
        let mut addr = match tls_info.addr().clone() {
            ConnectionAddr::TcpTls {
                host,
                port,
                tls_params,
                ..
            } => ConnectionAddr::TcpTls {
                host,
                port,
                insecure: !self.verify_certificate,
                tls_params,
            },
            addr => addr,
        };
        addr.set_danger_accept_invalid_hostnames(!self.check_hostname);
        Ok(tls_info.set_addr(addr))
    }
}

fn with_system_roots(configured: Vec<u8>) -> Vec<u8> {
    let system = rustls_native_certs::load_native_certs();
    let system_pem = system
        .certs
        .iter()
        .map(|cert| pem::encode(&pem::Pem::new("CERTIFICATE", cert.to_vec())));
    configured
        .into_iter()
        .chain(b"\n".iter().copied())
        .chain(system_pem.flat_map(String::into_bytes))
        .collect()
}

fn read(path: &PathBuf) -> Result<Vec<u8>, ConnectError> {
    std::fs::read(path).map_err(|source| ConnectError::TlsFile {
        path: path.display().to_string(),
        source,
    })
}

pub(crate) fn addr(host: &str, port: u16, tls: Option<&TlsOptions>) -> ConnectionAddr {
    match tls {
        Some(tls) => ConnectionAddr::TcpTls {
            host: host.to_owned(),
            port,
            insecure: !tls.verify_certificate,
            tls_params: None,
        },
        None => ConnectionAddr::Tcp(host.to_owned(), port),
    }
}

impl ClusterEndpoint {
    pub(crate) fn addrs(&self) -> impl Iterator<Item = ConnectionAddr> + '_ {
        self.nodes
            .iter()
            .map(|node| addr(&node.host, node.port, self.tls.as_ref()))
    }
}

impl Endpoint {
    pub(crate) fn connection_info(
        &self,
        credentials: &Credentials,
    ) -> Result<ConnectionInfo, ConnectError> {
        let (info, tls) = match self {
            Self::Url { url, tls } => (url.expose().into_connection_info()?, Some(tls)),
            Self::Host(host) => {
                let info = addr(&host.host, host.port, host.tls.as_ref()).into_connection_info()?;
                let settings = info.redis_settings().clone().set_db(host.db);
                (info.set_redis_settings(settings), host.tls.as_ref())
            }
            Self::Cluster(_) | Self::Sentinel(_) => {
                unreachable!("cluster and sentinel endpoints build their own connection info")
            }
        };
        let info = with_static_credentials(info, credentials);
        match (tls, info.addr()) {
            (Some(tls), ConnectionAddr::TcpTls { .. }) => tls.apply(info),
            _ => Ok(info),
        }
    }
}

fn with_static_credentials(info: ConnectionInfo, credentials: &Credentials) -> ConnectionInfo {
    let Credentials { username, password } = credentials;
    let settings = info.redis_settings().clone();
    let settings = match (username, settings.username()) {
        (Some(username), None) => settings.set_username(username.expose()),
        _ => settings,
    };
    let settings: RedisConnectionInfo = match (password, settings.password()) {
        (Some(password), None) => settings.set_password(password.expose()),
        _ => settings,
    };
    info.set_redis_settings(settings)
}
