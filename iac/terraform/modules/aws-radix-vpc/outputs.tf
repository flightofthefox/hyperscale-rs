output "vpc_id" {
  value = aws_vpc.vpc.id
}
output "public_subnet_id" {
  value = aws_subnet.public_subnet.id
}

output "additional_public_subnets" {
  value = {
    for key, val in aws_subnet.additional_public_subnet : key => val.id
  }
}
output "sg_allow_ssh_https_id" {
  value = aws_security_group.allow-ssh-http-https.id
}

output "sg_allow_gossip_port_id" {
  value = aws_security_group.allow-gossip-port.id
}

output "sg_core_security_group" {
  value = aws_security_group.core_security_group.id
}

module "public_validator_instances" {
  source          = "./tags-vars"
  nodes_with_tags = aws_eip.public_validator_instances
}

output "public_validator_instance_ips" {
  value      = module.public_validator_instances.out-vars-from-tags
  depends_on = [aws_eip.public_validator_instances]
}


module "public_fullnode_instances" {
  source          = "./tags-vars"
  nodes_with_tags = aws_eip.public_fullnode_instances
}

output "public_fullnode_instance_ips" {
  value      = module.public_fullnode_instances.out-vars-from-tags
  depends_on = [aws_eip.public_fullnode_instances]
}

module "public_fullnode_instances_archive_nodes" {
  source          = "./tags-vars"
  nodes_with_tags = aws_eip.public_fullnode_instances
}

output "public_non_archive_instance_ips" {
  value = {
    for key, val in module.public_fullnode_instances.out-vars-from-tags : key => val
    if val.non_archive_node == "true"
  }
  depends_on = [aws_eip.public_fullnode_instances]
}

output "gateway_node_instance_ips" {
  value = {
    for key, val in module.public_fullnode_instances.out-vars-from-tags : key => val
    if val.enable_transactions_api == "true"
  }
  depends_on = [aws_eip.public_fullnode_instances]
}
output "public_witnessnode_instance_ips" {
  value = {
    for key, val in aws_eip.public_witnessnode_instances : key => {
      public_ip = val.public_ip
    }
  }
  depends_on = [aws_eip.public_witnessnode_instances]
}

####################################################################
# CASSANDRA OUTPUTS
####################################################################

module "public_bootstrap_instances" {
  source          = "./tags-vars"
  nodes_with_tags = aws_eip.public_bootstrap_instances
}

output "public_bootstrap_instance_ips" {
  value      = module.public_bootstrap_instances.out-vars-from-tags
  depends_on = [aws_eip.public_bootstrap_instances]
}

module "public_spam_instances" {
  source          = "./tags-vars"
  nodes_with_tags = aws_eip.public_spam_instances
}

output "public_spam_instance_ips" {
  value      = module.public_spam_instances.out-vars-from-tags
  depends_on = [aws_eip.public_spam_instances]
}