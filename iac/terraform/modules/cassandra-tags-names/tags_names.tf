variable "bootstrap_nodes" {
  default = {}
}

variable "spam_nodes" {
  default = {}
}

variable "validator_nodes" {
  default = {}
}

variable "region" {
}

locals {
  BOOTSTRAP_LIST = flatten([for bootstrap, node_types in var.bootstrap_nodes :
    flatten([for node_type, nodes in node_types :
      flatten([for node, values in nodes :
        {
          tag                     = "${replace(var.region, "-", "_")}_${node_type}${bootstrap}"
          enable_faucet           = lookup(values, "enable_faucet", null)
          enable_health           = lookup(values, "enable_health", null)
          enable_metrics          = lookup(values, "enable_metrics", null)
          enable_construct        = lookup(values, "enable_construct", null)
          enable_system           = lookup(values, "enable_system", null)
          backup_ledger_node      = lookup(values, "backup_ledger_node", null)
          enable_validation       = lookup(values, "enable_validation", null)
          enable_version          = lookup(values, "enable_version", null)
          enable_universe_api     = lookup(values, "enable_universe_api", null)
          enable_chaos_api        = lookup(values, "enable_chaos_api", null)
          enable_transactions_api = lookup(values, "enable_transactions_api", null)
          non_archive_node        = lookup(values, "non_archive_node", null)
          latest_rc_code          = lookup(values, "latest_rc_code", null)
          enable_jmx_exporter     = lookup(values, "enable_jmx_exporter", null)
          internal_archive        = lookup(values, "internal_archive", null)
          extra_archive           = lookup(values, "extra_archive", null)
          dns_subdomain           = lookup(values, "dns_subdomain", null)
          explicit_instance_type  = lookup(values, "explicit_instance_type", null)
          explicit_ami            = lookup(values, "explicit_ami", null)
          genesis_validator       = lookup(values, "genesis_validator", null)
          access_type             = lookup(values, "access_type", null)
          collect_metrics         = lookup(values, "collect_metrics", null)
          collect_logs            = lookup(values, "collect_logs", null)
        }
      ])
      if node_type == "bootstrap"
    ])
  ])

  BOOTSTRAP_TAGS = {
    for key, values in local.BOOTSTRAP_LIST :
    values["tag"] => {
      core_private_ip         = lookup(values, "core_private_ip", null)
      enable_faucet           = lookup(values, "enable_faucet", null)
      enable_health           = lookup(values, "enable_health", null)
      enable_metrics          = lookup(values, "enable_metrics", null)
      collect_metrics         = lookup(values, "collect_metrics", null)
      collect_logs            = lookup(values, "collect_logs", null)
      enable_construct        = lookup(values, "enable_construct", null)
      enable_system           = lookup(values, "enable_system", null)
      enable_validation       = lookup(values, "enable_validation", null)
      enable_version          = lookup(values, "enable_version", null)
      backup_ledger_node      = lookup(values, "backup_ledger_node", null)
      enable_universe_api     = lookup(values, "enable_universe_api", null)
      enable_chaos_api        = lookup(values, "enable_chaos_api", null)
      enable_transactions_api = lookup(values, "enable_transactions_api", null)
      non_archive_node        = lookup(values, "non_archive_node", null)
      latest_rc_code          = lookup(values, "latest_rc_code", null)
      enable_jmx_exporter     = lookup(values, "enable_jmx_exporter", null)
      internal_archive        = lookup(values, "internal_archive", null)
      extra_archive           = lookup(values, "extra_archive", null)
      dns_subdomain           = lookup(values, "dns_subdomain", null)
      explicit_instance_type  = lookup(values, "explicit_instance_type", null)
      explicit_ami            = lookup(values, "explicit_ami", null)
      genesis_validator       = lookup(values, "genesis_validator", null)
    }
  }

  SPAM_LIST = flatten([for spam, node_types in var.spam_nodes :
    flatten([for node_type, nodes in node_types :
      flatten([for node, values in nodes :
        {
          tag                     = "${replace(var.region, "-", "_")}_${node_type}${spam}"
          enable_faucet           = lookup(values, "enable_faucet", null)
          enable_health           = lookup(values, "enable_health", null)
          enable_metrics          = lookup(values, "enable_metrics", null)
          enable_construct        = lookup(values, "enable_construct", null)
          enable_system           = lookup(values, "enable_system", null)
          enable_validation       = lookup(values, "enable_validation", null)
          backup_ledger_node      = lookup(values, "backup_ledger_node", null)
          enable_version          = lookup(values, "enable_version", null)
          enable_transactions_api = lookup(values, "enable_transactions_api", null)
          enable_universe_api     = lookup(values, "enable_universe_api", null)
          enable_chaos_api        = lookup(values, "enable_chaos_api", null)
          non_archive_node        = lookup(values, "non_archive_node", null)
          enable_jmx_exporter     = lookup(values, "enable_jmx_exporter", null)
          internal_archive        = lookup(values, "internal_archive", null)
          extra_archive           = lookup(values, "extra_archive", null)
          dns_subdomain           = lookup(values, "dns_subdomain", null)
          explicit_instance_type  = lookup(values, "explicit_instance_type", null)
          explicit_ami            = lookup(values, "explicit_ami", null)
          migration_aux_node      = lookup(values, "migration_aux_node", null)
          access_type             = lookup(values, "access_type", null)
          collect_metrics         = lookup(values, "collect_metrics", null)
          collect_logs            = lookup(values, "collect_logs", null)
        }
      ])
      if node_type == "spam"
    ])
  ])

  SPAM_TAGS = {
    for key, values in local.SPAM_LIST :
    values["tag"] => {
      enable_faucet           = lookup(values, "enable_faucet", null)
      enable_health           = lookup(values, "enable_health", null)
      enable_metrics          = lookup(values, "enable_metrics", null)
      enable_construct        = lookup(values, "enable_construct", null)
      backup_ledger_node      = lookup(values, "backup_ledger_node", null)
      enable_system           = lookup(values, "enable_system", null)
      enable_validation       = lookup(values, "enable_validation", null)
      enable_version          = lookup(values, "enable_version", null)
      enable_transactions_api = lookup(values, "enable_transactions_api", null)
      non_archive_node        = lookup(values, "non_archive_node", null)
      enable_universe_api     = lookup(values, "enable_universe_api", null)
      enable_chaos_api        = lookup(values, "enable_chaos_api", null)
      enable_jmx_exporter     = lookup(values, "enable_jmx_exporter", null)
      internal_archive        = lookup(values, "internal_archive", null)
      extra_archive           = lookup(values, "extra_archive", null)
      dns_subdomain           = lookup(values, "dns_subdomain", null)
      explicit_instance_type  = lookup(values, "explicit_instance_type", null)
      explicit_ami            = lookup(values, "explicit_ami", null)
      migration_aux_node      = lookup(values, "migration_aux_node", null)
      access_type             = lookup(values, "access_type", null)
      collect_metrics         = lookup(values, "collect_metrics", null)
      collect_logs            = lookup(values, "collect_logs", null)
    }
  }

  VALIDATOR_LIST = flatten([for validator, node_types in var.validator_nodes :
    flatten([for node_type, nodes in node_types :
      flatten([for node, values in nodes :
        {
          tag                     = "${replace(var.region, "-", "_")}_${node_type}${validator}"
          enable_faucet           = lookup(values, "enable_faucet", null)
          enable_health           = lookup(values, "enable_health", null)
          enable_metrics          = lookup(values, "enable_metrics", null)
          enable_construct        = lookup(values, "enable_construct", null)
          enable_system           = lookup(values, "enable_system", null)
          enable_validation       = lookup(values, "enable_validation", null)
          backup_ledger_node      = lookup(values, "backup_ledger_node", null)
          enable_version          = lookup(values, "enable_version", null)
          enable_transactions_api = lookup(values, "enable_transactions_api", null)
          enable_universe_api     = lookup(values, "enable_universe_api", null)
          enable_chaos_api        = lookup(values, "enable_chaos_api", null)
          non_archive_node        = lookup(values, "non_archive_node", null)
          enable_jmx_exporter     = lookup(values, "enable_jmx_exporter", null)
          internal_archive        = lookup(values, "internal_archive", null)
          extra_archive           = lookup(values, "extra_archive", null)
          dns_subdomain           = lookup(values, "dns_subdomain", null)
          explicit_instance_type  = lookup(values, "explicit_instance_type", null)
          explicit_ami            = lookup(values, "explicit_ami", null)
          migration_aux_node      = lookup(values, "migration_aux_node", null)
          access_type             = lookup(values, "access_type", null)
          collect_metrics         = lookup(values, "collect_metrics", null)
          collect_logs            = lookup(values, "collect_logs", null)
        }
      ])
      if node_type == "validator"
    ])
  ])

  VALIDATOR_TAGS = {
    for key, values in local.VALIDATOR_LIST :
    values["tag"] => {
      enable_faucet           = lookup(values, "enable_faucet", null)
      enable_health           = lookup(values, "enable_health", null)
      enable_metrics          = lookup(values, "enable_metrics", null)
      enable_construct        = lookup(values, "enable_construct", null)
      backup_ledger_node      = lookup(values, "backup_ledger_node", null)
      enable_system           = lookup(values, "enable_system", null)
      enable_validation       = lookup(values, "enable_validation", null)
      enable_version          = lookup(values, "enable_version", null)
      enable_transactions_api = lookup(values, "enable_transactions_api", null)
      non_archive_node        = lookup(values, "non_archive_node", null)
      enable_universe_api     = lookup(values, "enable_universe_api", null)
      enable_chaos_api        = lookup(values, "enable_chaos_api", null)
      enable_jmx_exporter     = lookup(values, "enable_jmx_exporter", null)
      internal_archive        = lookup(values, "internal_archive", null)
      extra_archive           = lookup(values, "extra_archive", null)
      dns_subdomain           = lookup(values, "dns_subdomain", null)
      explicit_instance_type  = lookup(values, "explicit_instance_type", null)
      explicit_ami            = lookup(values, "explicit_ami", null)
      migration_aux_node      = lookup(values, "migration_aux_node", null)
      access_type             = lookup(values, "access_type", null)
      collect_metrics         = lookup(values, "collect_metrics", null)
      collect_logs            = lookup(values, "collect_logs", null)
    }
  }

}

output "BOOTSTRAP_TAGS" {
  value = local.BOOTSTRAP_TAGS
}

output "SPAM_TAGS" {
  value = local.SPAM_TAGS
}

output "VALIDATOR_TAGS" {
  value = local.VALIDATOR_TAGS
}
