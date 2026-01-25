//! # Bridge Data Plane Entry Point
//!
//! Main executable that initializes the bridge and starts the gRPC server
//! for receiving rule installation requests from the control plane.

use bridge::{grpc_server::start_grpc_server, Bridge};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=================================================");
    println!("  TUPL Data Plane - Bridge & Rule Installation  ");
    println!("=================================================");
    println!();
    
    println!("Initializing Bridge with tiered storage...");
    let bridge_inst = match Bridge::init() {
        Ok(bridge) => Arc::new(bridge),
        Err(e) => {
            eprintln!("✗ Failed to initialize Bridge: {}", e);
            return Err(e.into());
        }
    };
    println!("✓ Bridge initialized");
    println!("  - {} rules loaded", bridge_inst.rule_count());
    println!("  - Version: {}", bridge_inst.version());
    println!();

    // Start gRPC server for rule installation and enforcement
    let grpc_port = 50051;
    let management_plane_url = std::env::var("MANAGEMENT_PLANE_URL")
        .unwrap_or_else(|_| "http://localhost:8000/api/v1".to_string());

    println!("Starting gRPC server for rule installation and enforcement...");
    println!("  - Listening on: 0.0.0.0:{}", grpc_port);
    println!("  - Management Plane: {}", management_plane_url);
    println!();

    // Start the gRPC server
    start_grpc_server(bridge_inst, grpc_port, management_plane_url).await?;

    println!("=================================================");
    println!("  Data Plane Shut Down");
    println!("=================================================");

    Ok(())
}
