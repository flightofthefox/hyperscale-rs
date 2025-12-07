//! End-to-end tests for the production runner.
//!
//! These tests validate the production runner with real localhost QUIC networking.
//! All tests use `#[serial]` to avoid port conflicts and state leakage.

mod fixtures;

use hyperscale_bft::BftConfig;
use hyperscale_production::{
    ProductionRunner, RocksDbStorage, ThreadPoolConfig, ThreadPoolManager,
};
use hyperscale_types::{
    Block, BlockHeader, BlockHeight, Hash, QuorumCertificate, ShardGroupId, ValidatorId,
};
use fixtures::TestFixtures;
use serial_test::serial;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::info;

/// Test timeout values (from design spec).
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(5);
#[allow(dead_code)]
const SINGLE_BLOCK_TIMEOUT: Duration = Duration::from_secs(10);
#[allow(dead_code)]
const SYNC_CATCH_UP_TIMEOUT: Duration = Duration::from_secs(30);
#[allow(dead_code)]
const OVERALL_TEST_TIMEOUT: Duration = Duration::from_secs(60);

// ============================================================================
// Single Node Tests (no real network)
// ============================================================================

#[tokio::test]
#[serial]
async fn test_production_runner_starts() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    let fixtures = TestFixtures::new(42, 1);
    let thread_pools = Arc::new(ThreadPoolManager::auto().unwrap());

    let runner = ProductionRunner::with_thread_pools(
        0,
        fixtures.topology(0),
        fixtures.signing_key(0),
        BftConfig::default(),
        1000,
        thread_pools,
    )
    .unwrap();

    // Verify basic state
    assert_eq!(runner.local_shard(), ShardGroupId(0));
    assert!(runner.network().is_none()); // No network configured
    assert!(!runner.is_syncing());

    info!("Production runner created successfully");
}

#[tokio::test]
#[serial]
async fn test_runner_starts_and_stops() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    let fixtures = TestFixtures::new(42, 1);
    let thread_pools = Arc::new(ThreadPoolManager::auto().unwrap());

    let mut runner = ProductionRunner::with_thread_pools(
        0,
        fixtures.topology(0),
        fixtures.signing_key(0),
        BftConfig::default(),
        1000,
        thread_pools,
    )
    .unwrap();

    // Get the shutdown handle before running
    let shutdown = runner
        .shutdown_handle()
        .expect("Should have shutdown handle");

    // Spawn the runner
    let handle = tokio::spawn(runner.run());

    // Give it a moment to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Gracefully shutdown
    drop(shutdown);

    // Wait for runner to exit
    let result = timeout(Duration::from_secs(5), handle).await;
    assert!(result.is_ok(), "Runner should exit cleanly");
    assert!(result.unwrap().is_ok(), "Runner should return Ok");

    info!("Runner start/stop test completed");
}

#[tokio::test]
#[serial]
async fn test_crypto_verification_thread_pool() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    let fixtures = TestFixtures::new(42, 1);
    let config = ThreadPoolConfig::builder()
        .crypto_threads(2)
        .execution_threads(2)
        .io_threads(1)
        .build()
        .unwrap();

    let thread_pools = Arc::new(ThreadPoolManager::new(config).unwrap());

    let runner = ProductionRunner::with_thread_pools(
        0,
        fixtures.topology(0),
        fixtures.signing_key(0),
        BftConfig::default(),
        1000,
        thread_pools.clone(),
    )
    .unwrap();

    // Verify thread pool configuration
    let pool_config = runner.thread_pools().config();
    assert_eq!(pool_config.crypto_threads, 2);
    assert_eq!(pool_config.execution_threads, 2);
    assert_eq!(pool_config.io_threads, 1);

    info!("Thread pool configuration verified");
}

#[tokio::test]
#[serial]
async fn test_storage_operations() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test_db");

    let storage = RocksDbStorage::open(&db_path).unwrap();

    // Test block storage
    let header = BlockHeader {
        height: BlockHeight(1),
        parent_hash: Hash::from_bytes(&[0u8; 32]),
        parent_qc: QuorumCertificate::genesis(),
        proposer: ValidatorId(0),
        timestamp: 1000,
        round: 1,
        is_fallback: false,
    };

    let block = Block {
        header: header.clone(),
        transactions: vec![],
        committed_certificates: vec![],
        deferred: vec![],
        aborted: vec![],
    };

    let qc = QuorumCertificate::genesis();

    storage.put_block(BlockHeight(1), &block, &qc);

    // Retrieve the block
    let retrieved = storage.get_block(BlockHeight(1));
    assert!(retrieved.is_some());
    let (retrieved_block, _retrieved_qc) = retrieved.unwrap();
    assert_eq!(retrieved_block.header.height, BlockHeight(1));

    // Test chain metadata
    storage.set_chain_metadata(
        BlockHeight(1),
        Some(Hash::from_bytes(&[1u8; 32])),
        Some(&qc),
    );
    let (height, hash, _) = storage.get_chain_metadata();
    assert_eq!(height, BlockHeight(1));
    assert!(hash.is_some());

    info!("Storage operations verified");
}

// ============================================================================
// Network Tests (localhost QUIC)
// ============================================================================

#[tokio::test]
#[serial]
async fn test_network_adapter_starts() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    use hyperscale_production::network::{Libp2pAdapter, Libp2pConfig};
    use libp2p::identity;

    let keypair = identity::Keypair::generate_ed25519();
    let validator_id = ValidatorId(0);
    let shard = ShardGroupId(0);

    // Use port 0 for OS-assigned port
    let config = Libp2pConfig {
        listen_addresses: vec!["/ip4/127.0.0.1/udp/0/quic-v1".parse().unwrap()],
        bootstrap_peers: vec![],
        ..Default::default()
    };

    let (event_tx, _event_rx) = mpsc::channel(100);

    let result = timeout(
        CONNECTION_TIMEOUT,
        Libp2pAdapter::new(config, keypair, validator_id, shard, event_tx),
    )
    .await;

    assert!(result.is_ok(), "Adapter creation should not timeout");
    let (adapter, _sync_rx) = result.unwrap().unwrap();

    // Verify adapter state
    assert_eq!(adapter.local_validator_id(), validator_id);

    // Get listen addresses (should have at least one after initialization)
    tokio::time::sleep(Duration::from_millis(100)).await;
    let addrs = adapter.listen_addresses().await;
    info!(addresses = ?addrs, "Adapter listening on");

    info!("Network adapter started successfully");
}

#[tokio::test]
#[serial]
async fn test_two_node_connection() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    use hyperscale_production::network::{Libp2pAdapter, Libp2pConfig};
    use libp2p::identity;

    // Node 1
    let keypair1 = identity::Keypair::generate_ed25519();
    let config1 = Libp2pConfig {
        listen_addresses: vec!["/ip4/127.0.0.1/udp/0/quic-v1".parse().unwrap()],
        bootstrap_peers: vec![],
        ..Default::default()
    };
    let (event_tx1, _event_rx1) = mpsc::channel(100);

    let (adapter1, _sync_rx1) = Libp2pAdapter::new(
        config1,
        keypair1,
        ValidatorId(0),
        ShardGroupId(0),
        event_tx1,
    )
    .await
    .unwrap();

    // Wait for node 1 to be ready and get its address
    tokio::time::sleep(Duration::from_millis(200)).await;
    let addrs1 = adapter1.listen_addresses().await;
    assert!(!addrs1.is_empty(), "Node 1 should have listen addresses");
    let node1_addr = addrs1[0].clone();
    info!(addr = %node1_addr, "Node 1 listening");

    // Node 2 - bootstrap to node 1
    let keypair2 = identity::Keypair::generate_ed25519();
    let config2 = Libp2pConfig {
        listen_addresses: vec!["/ip4/127.0.0.1/udp/0/quic-v1".parse().unwrap()],
        bootstrap_peers: vec![node1_addr.clone()],
        ..Default::default()
    };
    let (event_tx2, _event_rx2) = mpsc::channel(100);

    let (adapter2, _sync_rx2) = Libp2pAdapter::new(
        config2,
        keypair2,
        ValidatorId(1),
        ShardGroupId(0),
        event_tx2,
    )
    .await
    .unwrap();

    // Wait for connection to establish
    let connected = timeout(CONNECTION_TIMEOUT, async {
        loop {
            let peers1 = adapter1.connected_peers().await;
            let peers2 = adapter2.connected_peers().await;

            if !peers1.is_empty() && !peers2.is_empty() {
                return (peers1, peers2);
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await;

    assert!(connected.is_ok(), "Nodes should connect within timeout");
    let (peers1, peers2) = connected.unwrap();

    info!(
        node1_peers = peers1.len(),
        node2_peers = peers2.len(),
        "Nodes connected"
    );

    assert!(!peers1.is_empty(), "Node 1 should have peers");
    assert!(!peers2.is_empty(), "Node 2 should have peers");
}

#[tokio::test]
#[serial]
async fn test_topic_subscription() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    use hyperscale_production::network::{Libp2pAdapter, Libp2pConfig};
    use libp2p::identity;

    let keypair = identity::Keypair::generate_ed25519();
    let config = Libp2pConfig {
        listen_addresses: vec!["/ip4/127.0.0.1/udp/0/quic-v1".parse().unwrap()],
        bootstrap_peers: vec![],
        ..Default::default()
    };
    let (event_tx, _event_rx) = mpsc::channel(100);

    let (adapter, _sync_rx) =
        Libp2pAdapter::new(config, keypair, ValidatorId(0), ShardGroupId(0), event_tx)
            .await
            .unwrap();

    // Subscribe to shard topics
    let result = adapter.subscribe_shard(ShardGroupId(0)).await;
    assert!(result.is_ok(), "Should subscribe to shard topics");

    info!("Topic subscription successful");
}

// ============================================================================
// Production Runner with Network Tests
// ============================================================================

#[tokio::test]
#[serial]
async fn test_production_runner_with_network() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    use hyperscale_production::network::Libp2pConfig;
    use libp2p::identity;

    let fixtures = TestFixtures::new(42, 1);
    let thread_pools = Arc::new(ThreadPoolManager::auto().unwrap());

    let ed25519_keypair = identity::Keypair::generate_ed25519();
    let network_config = Libp2pConfig {
        listen_addresses: vec!["/ip4/127.0.0.1/udp/0/quic-v1".parse().unwrap()],
        bootstrap_peers: vec![],
        ..Default::default()
    };

    let runner = timeout(
        CONNECTION_TIMEOUT,
        ProductionRunner::with_network(
            0,
            fixtures.topology(0),
            fixtures.signing_key(0),
            BftConfig::default(),
            1000,
            thread_pools,
            network_config,
            ed25519_keypair,
            None, // No storage for this test
        ),
    )
    .await;

    assert!(runner.is_ok(), "Runner creation should not timeout");
    let mut runner = runner.unwrap().unwrap();

    // Verify network is configured
    assert!(runner.network().is_some(), "Network should be configured");

    let network = runner.network().unwrap();
    info!(peer_id = %network.local_peer_id(), "Runner has network");

    // Get listen addresses
    tokio::time::sleep(Duration::from_millis(100)).await;
    let addrs = network.listen_addresses().await;
    info!(addresses = ?addrs, "Runner listening on");

    // Get shutdown handle before running
    let shutdown = runner
        .shutdown_handle()
        .expect("Should have shutdown handle");
    let handle = tokio::spawn(runner.run());

    tokio::time::sleep(Duration::from_millis(500)).await;
    drop(shutdown);

    let result = timeout(Duration::from_secs(5), handle).await;
    assert!(result.is_ok(), "Runner should exit cleanly");

    info!("Production runner with network test completed");
}

// ============================================================================
// Graceful Shutdown Tests
// ============================================================================

#[tokio::test]
#[serial]
async fn test_graceful_shutdown() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    use hyperscale_production::network::Libp2pConfig;
    use libp2p::identity;

    let fixtures = TestFixtures::new(42, 1);
    let thread_pools = Arc::new(ThreadPoolManager::auto().unwrap());

    let ed25519_keypair = identity::Keypair::generate_ed25519();
    let network_config = Libp2pConfig {
        listen_addresses: vec!["/ip4/127.0.0.1/udp/0/quic-v1".parse().unwrap()],
        bootstrap_peers: vec![],
        ..Default::default()
    };

    let mut runner = ProductionRunner::with_network(
        0,
        fixtures.topology(0),
        fixtures.signing_key(0),
        BftConfig::default(),
        1000,
        thread_pools,
        network_config,
        ed25519_keypair,
        None,
    )
    .await
    .unwrap();

    let shutdown = runner
        .shutdown_handle()
        .expect("Should have shutdown handle");
    let handle = tokio::spawn(runner.run());

    // Let it run briefly
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Shutdown via handle
    drop(shutdown);

    // Should exit within 5 seconds (graceful shutdown max)
    let result = timeout(Duration::from_secs(5), handle).await;
    assert!(
        result.is_ok(),
        "Runner should exit within graceful shutdown timeout"
    );

    let run_result = result.unwrap();
    assert!(run_result.is_ok(), "Runner should return Ok on shutdown");

    info!("Graceful shutdown test completed");
}
