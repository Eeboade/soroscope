//! Differential resource checks for local Soroban execution versus RPC simulation.

use crate::runner::{ContractInvocation, LocalRunner};
use crate::simulation::{SimulationEngine, SimulationError, SimulationResult, SorobanResources};
use serde::Deserialize;
use std::sync::Arc;

const TOLERANCES_JSON: &str = include_str!("../config/differential-tolerances.json");

#[derive(Debug, Clone, Deserialize)]
struct ToleranceFile {
    protocol_snapshots: std::collections::HashMap<String, MeterTolerances>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MeterTolerances {
    /// Relative CPU tolerance in basis points (1000 = 10%).
    pub cpu_relative_error_bps: u32,
    /// Relative memory tolerance in basis points (1000 = 10%).
    pub memory_relative_error_bps: u32,
}

impl MeterTolerances {
    pub fn for_protocol(protocol_version: u32) -> Result<Self, SimulationError> {
        let fixtures: ToleranceFile =
            serde_json::from_str(TOLERANCES_JSON).map_err(SimulationError::SerializationError)?;
        fixtures
            .protocol_snapshots
            .get(&protocol_version.to_string())
            .cloned()
            .ok_or_else(|| {
                SimulationError::InvalidContract(format!(
                    "No differential tolerances checked in for protocol {protocol_version}"
                ))
            })
    }
}

#[derive(Debug, Clone)]
pub struct DifferentialInput {
    pub wasm: Vec<u8>,
    pub contract_id: String,
    pub function_name: String,
    pub args: Vec<String>,
    pub protocol_version: u32,
    /// Set only when both executions are known to have used the exact same ledger entries.
    pub same_ledger_entries_confirmed: bool,
}

#[derive(Debug, Clone)]
pub struct DifferentialSide {
    pub resources: Option<SorobanResources>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DifferentialReport {
    pub protocol_version: u32,
    pub local: DifferentialSide,
    pub rpc: DifferentialSide,
    pub cpu_relative_error_bps: Option<u64>,
    pub memory_relative_error_bps: Option<u64>,
    pub ledger_read_bytes_match: Option<bool>,
    pub ledger_write_bytes_match: Option<bool>,
    pub passed: bool,
    pub failures: Vec<String>,
}

/// Runs both simulation paths independently; neither path is used as a fallback for the other.
pub struct DifferentialRunner {
    local: Arc<LocalRunner>,
    rpc: SimulationEngine,
}

impl DifferentialRunner {
    pub fn new(rpc_url: String, local: Arc<LocalRunner>) -> Self {
        Self {
            local,
            rpc: SimulationEngine::new(rpc_url),
        }
    }

    pub async fn run(
        &self,
        input: DifferentialInput,
    ) -> Result<DifferentialReport, SimulationError> {
        let (tolerances, tolerance_error) =
            match MeterTolerances::for_protocol(input.protocol_version) {
                Ok(tolerances) => (tolerances, None),
                Err(error) => (
                    MeterTolerances {
                        cpu_relative_error_bps: 0,
                        memory_relative_error_bps: 0,
                    },
                    Some(error.to_string()),
                ),
            };
        let local_result = self
            .run_local(&input)
            .await
            .map(|result| result.resources)
            .map_err(|error| error.to_string());

        // Always attempt RPC, even when local execution failed, so the report retains both sides.
        let rpc_result = self
            .rpc
            .simulate_from_contract_id(
                &input.contract_id,
                &input.function_name,
                input.args.clone(),
                None,
                Some(input.protocol_version),
                None,
            )
            .await
            .map(|result| result.resources)
            .map_err(|error| error.to_string());

        let mut report = compare_results(
            input.protocol_version,
            tolerances,
            input.same_ledger_entries_confirmed,
            local_result,
            rpc_result,
        );
        if let Some(error) = tolerance_error {
            report
                .failures
                .push(format!("protocol tolerance unavailable: {error}"));
            report.passed = false;
        }
        Ok(report)
    }

    async fn run_local(
        &self,
        input: &DifferentialInput,
    ) -> Result<SimulationResult, SimulationError> {
        let contract_hash = self.rpc.parse_contract_id(&input.contract_id)?;
        self.local
            .load_wasm(contract_hash, input.wasm.clone())
            .await;
        let invocation = ContractInvocation::new(
            contract_hash,
            input.function_name.clone(),
            input.args.clone(),
        );
        self.local
            .simulate_with_protocol(&invocation, Some(input.protocol_version))
            .await
    }
}

fn compare_results(
    protocol_version: u32,
    tolerances: MeterTolerances,
    same_ledger_entries_confirmed: bool,
    local_result: Result<SorobanResources, String>,
    rpc_result: Result<SorobanResources, String>,
) -> DifferentialReport {
    let local = side(local_result);
    let rpc = side(rpc_result);
    let mut failures = Vec::new();

    if let Some(error) = &local.error {
        failures.push(format!("local simulation failed: {error}"));
    }
    if let Some(error) = &rpc.error {
        failures.push(format!("RPC simulation failed: {error}"));
    }

    let (
        cpu_relative_error_bps,
        memory_relative_error_bps,
        ledger_read_bytes_match,
        ledger_write_bytes_match,
    ) = match (&local.resources, &rpc.resources) {
        (Some(local_resources), Some(rpc_resources)) => {
            let cpu_error = relative_error_bps(
                local_resources.cpu_instructions,
                rpc_resources.cpu_instructions,
            );
            let memory_error =
                relative_error_bps(local_resources.ram_bytes, rpc_resources.ram_bytes);

            if cpu_error > u64::from(tolerances.cpu_relative_error_bps) {
                failures.push(format!(
                        "CPU relative error {cpu_error} bps exceeds protocol {protocol_version} tolerance {} bps",
                        tolerances.cpu_relative_error_bps
                    ));
            }
            if memory_error > u64::from(tolerances.memory_relative_error_bps) {
                failures.push(format!(
                        "memory relative error {memory_error} bps exceeds protocol {protocol_version} tolerance {} bps",
                        tolerances.memory_relative_error_bps
                    ));
            }

            let read_match = same_ledger_entries_confirmed
                .then_some(local_resources.ledger_read_bytes == rpc_resources.ledger_read_bytes);
            let write_match = same_ledger_entries_confirmed
                .then_some(local_resources.ledger_write_bytes == rpc_resources.ledger_write_bytes);
            if read_match == Some(false) {
                failures
                    .push("ledger_read_bytes must match for identical ledger entries".to_string());
            }
            if write_match == Some(false) {
                failures
                    .push("ledger_write_bytes must match for identical ledger entries".to_string());
            }

            (Some(cpu_error), Some(memory_error), read_match, write_match)
        }
        _ => (None, None, None, None),
    };

    DifferentialReport {
        protocol_version,
        local,
        rpc,
        cpu_relative_error_bps,
        memory_relative_error_bps,
        ledger_read_bytes_match,
        ledger_write_bytes_match,
        passed: failures.is_empty(),
        failures,
    }
}

fn side(result: Result<SorobanResources, String>) -> DifferentialSide {
    match result {
        Ok(resources) => DifferentialSide {
            resources: Some(resources),
            error: None,
        },
        Err(error) => DifferentialSide {
            resources: None,
            error: Some(error),
        },
    }
}

fn relative_error_bps(local: u64, rpc: u64) -> u64 {
    if rpc == 0 {
        return if local == 0 { 0 } else { u64::MAX };
    }
    let difference = local.abs_diff(rpc) as u128;
    ((difference * 10_000) / u128::from(rpc)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resources(cpu: u64, memory: u64) -> SorobanResources {
        SorobanResources {
            cpu_instructions: cpu,
            ram_bytes: memory,
            ..SorobanResources::default()
        }
    }

    #[test]
    fn mock_rpc_cpu_mismatch_over_tolerance_names_the_meter() {
        let report = compare_results(
            22,
            MeterTolerances {
                cpu_relative_error_bps: 1000,
                memory_relative_error_bps: 1000,
            },
            false,
            Ok(resources(115, 100)),
            Ok(resources(100, 100)),
        );

        assert!(!report.passed);
        assert_eq!(report.cpu_relative_error_bps, Some(1500));
        assert!(report
            .failures
            .iter()
            .any(|failure| failure.contains("CPU")));
        assert!(report.local.resources.is_some());
        assert!(report.rpc.resources.is_some());
    }

    #[test]
    fn identical_mock_fixtures_pass_and_require_exact_ledger_bytes() {
        let mut local = resources(100, 200);
        local.ledger_read_bytes = 64;
        local.ledger_write_bytes = 32;
        let mut rpc = local.clone();
        rpc.transaction_size_bytes = 999;

        let report = compare_results(
            22,
            MeterTolerances {
                cpu_relative_error_bps: 1000,
                memory_relative_error_bps: 1000,
            },
            true,
            Ok(local),
            Ok(rpc),
        );

        assert!(report.passed, "{:?}", report.failures);
        assert_eq!(report.ledger_read_bytes_match, Some(true));
        assert_eq!(report.ledger_write_bytes_match, Some(true));
    }

    #[test]
    fn both_side_results_are_kept_when_local_execution_fails() {
        let report = compare_results(
            22,
            MeterTolerances {
                cpu_relative_error_bps: 1000,
                memory_relative_error_bps: 1000,
            },
            false,
            Err("local unavailable".to_string()),
            Ok(resources(100, 200)),
        );

        assert!(!report.passed);
        assert!(report.local.error.is_some());
        assert!(report.rpc.resources.is_some());
    }
}
