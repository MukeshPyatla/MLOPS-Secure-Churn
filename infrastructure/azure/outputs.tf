output "resource_group_name" {
  value = azurerm_resource_group.rg.name
}

output "aml_workspace_name" {
  value = azurerm_machine_learning_workspace.aml.name
}

output "storage_account_name" {
  value = azurerm_storage_account.storage.name
}

output "databricks_workspace_url" {
  value = azurerm_databricks_workspace.databricks.workspace_url
}