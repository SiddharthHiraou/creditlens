variable "name" {
  description = "Name prefix for every resource."
  type        = string
  default     = "creditlens"
}

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name. 'prod' enables deletion protection and longer backups."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be dev, staging or prod."
  }
}

variable "image_tag" {
  description = "ECR image tag to deploy. Tags are immutable, so this pins an exact build."
  type        = string
  default     = "latest"
}

variable "db_instance_class" {
  description = "RDS instance class."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_password" {
  description = <<-EOT
    Postgres password.

    Passed as a variable so nothing is committed. In a real deployment this
    belongs in Secrets Manager and is injected into the task definition by ARN
    rather than as a plaintext environment variable — see NOT_INCLUDED in main.tf.
  EOT
  type        = string
  sensitive   = true
}

variable "api_key" {
  description = "API key for the scoring endpoints. Same caveat as db_password."
  type        = string
  sensitive   = true
}

variable "task_cpu" {
  description = "Fargate CPU units. 1024 = 1 vCPU."
  type        = string
  default     = "1024"
}

variable "task_memory" {
  description = "Fargate memory in MiB. ONNX plus the TreeSHAP explainer needs headroom."
  type        = string
  default     = "2048"
}

variable "desired_count" {
  description = "Number of API tasks."
  type        = number
  default     = 2
}
