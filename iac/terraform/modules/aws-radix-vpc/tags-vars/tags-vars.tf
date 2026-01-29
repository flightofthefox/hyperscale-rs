variable "nodes_with_tags" {}
locals {

  vars_obj = {
    for key, val in var.nodes_with_tags :
    key => {
      public_ip       = val.public_ip
      core_private_ip = lookup(val.tags, "radixdlt:core-private-ip", null)
      enable_faucet   = lookup(val.tags, "radixdlt:faucet", null)
      enable_health : lookup(val.tags, "radixdlt:enable-health-api", null)
      enable_construct : lookup(val.tags, "radixdlt:enable-construct-api", null)
      enable_system : lookup(val.tags, "radixdlt:enable-system-api", null)
      enable_validation : lookup(val.tags, "radixdlt:enable-validation-api", null)
      enable_version : lookup(val.tags, "radixdlt:enable-version-api", null)
      enable_metrics : lookup(val.tags, "radixdlt:enable-metrics-api", null)
      enable_transactions_api = lookup(val.tags, "radixdlt:enable-transactions-api", null)
      non_archive_node        = lookup(val.tags, "radixdlt:non-archive-node", null)
      enable_universe_api     = lookup(val.tags, "radixdlt:enable-universe-api", null)
      enable_chaos_api        = lookup(val.tags, "radixdlt:enable-chaos-api", null)
      backup_ledger_node      = lookup(val.tags, "radixdlt:backup-ledger-node", null)
      latest_rc_code : lookup(val.tags, "radixdlt:latest-rc-code", null)
      enable_jmx_exporter    = lookup(val.tags, "radixdlt:enable-jmx-exporter", null)
      internal_archive       = lookup(val.tags, "radixdlt:internal-archive", null)
      extra_archive          = lookup(val.tags, "radixdlt:extra-archive", null)
      dns_subdomain          = lookup(val.tags, "radixdlt:dns-subdomain", null)
      explicit_instance_type = lookup(val.tags, "radixdlt:explicit-instance-type", null)
      explicit_ami           = lookup(val.tags, "radixdlt:explicit-ami", null)
      genesis_validator      = lookup(val.tags, "radixdlt:genesis-validator", null)
      migration_aux_node     = lookup(val.tags, "radixdlt:migration-aux-node", null)
      access_type            = lookup(val.tags, "radixdlt:access-type", null)
      collect_metrics        = lookup(val.tags, "radixdlt:collect-metrics", null)
      collect_logs           = lookup(val.tags, "radixdlt:collect-logs", null)
    }
  }

}


output "out-vars-from-tags" {
  value = local.vars_obj
}
