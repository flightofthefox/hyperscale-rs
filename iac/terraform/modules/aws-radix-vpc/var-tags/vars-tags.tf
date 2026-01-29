variable "nodes_with_vars" {}
locals {

  tags_obj = {
    for key, val in var.nodes_with_vars :
    key => {
      Name                       = key
      "radixdlt:core-private-ip" = lookup(val, "core_private_ip", null)
      "radixdlt:faucet"          = lookup(val, "enable_faucet", null)
      "radixdlt:enable-health-api" : lookup(val, "enable_health", null)
      "radixdlt:enable-construct-api" : lookup(val, "enable_construct", null)
      "radixdlt:enable-system-api" : lookup(val, "enable_system", null)
      "radixdlt:enable-validation-api" : lookup(val, "enable_validation", null)
      "radixdlt:enable-version-api" : lookup(val, "enable_version", null)
      "radixdlt:enable-metrics-api" : lookup(val, "enable_metrics", null)
      "radixdlt:enable-transactions-api" : lookup(val, "enable_transactions_api", null)
      "radixdlt:non-archive-node" : lookup(val, "non_archive_node", null)
      "radixdlt:enable-universe-api" : lookup(val, "enable_universe_api", null)
      "radixdlt:enable-chaos-api" : lookup(val, "enable_chaos_api", null)
      "radixdlt:backup-ledger-node" : lookup(val, "backup_ledger_node", null)
      "radixdlt:latest-rc-code" : lookup(val, "latest_rc_code", null)
      "radixdlt:enable-jmx-exporter" : lookup(val, "enable_jmx_exporter", null)
      "radixdlt:internal-archive" : lookup(val, "internal_archive", null)
      "radixdlt:extra-archive" : lookup(val, "extra_archive", null)
      "radixdlt:dns-subdomain" : lookup(val, "dns_subdomain", null)
      "radixdlt:explicit-instance-type" : lookup(val, "explicit_instance_type", null)
      "radixdlt:explicit-ami" : lookup(val, "explicit_ami", null)
      "radixdlt:genesis-validator" : lookup(val, "genesis_validator", null)
      "radixdlt:migration-aux-node" : lookup(val, "migration_aux_node", null)
      "radixdlt:access-type" : lookup(val, "access_type", null)
      "radixdlt:collect-metrics" : lookup(val, "collect_metrics", null)
      "radixdlt:collect-logs" : lookup(val, "collect_logs", null)
    }
  }

}

output "out-tags-from-vars" {
  value = local.tags_obj
}
