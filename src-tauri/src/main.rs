// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;
use fs_core::{FsCore, FileEntry};

// IPC Commands for frontend communication
#[tauri::command]
fn read_brick_file(path: String) -> Result<String, String> {
    FsCore::read_file(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn write_brick_file(path: String, content: String) -> Result<(), String> {
    FsCore::write_file(&path, content).map_err(|e| e.to_string())
}

#[tauri::command]
fn list_directory(path: String) -> Result<Vec<FileEntry>, String> {
    FsCore::list_dir(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn create_directory(path: String) -> Result<(), String> {
    FsCore::create_dir(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn delete_item(path: String) -> Result<(), String> {
    FsCore::delete(&path).map_err(|e| e.to_string())
}

#[tauri::command]
async fn compile_isa(source: String) -> Result<String, String> {
    // Placeholder for ISA compilation
    // This will be replaced with actual compiler kernel logic
    Ok(format!("Compiled {} bytes of ISA source", source.len()))
}

#[tauri::command]
fn get_system_info() -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "platform": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "kernel_version": kernel_bridge::KernelBridge::version(),
        "status": "READY"
    }))
}

mod menu;

fn main() {
    tauri::Builder::default()
        .menu(menu::get_menu())
        .on_menu_event(|event| {
            match event.menu_item_id() {
                "new_file" => {
                    event.window().emit("menu-event", "new-file").unwrap();
                }
                "open_file" => {
                    event.window().emit("menu-event", "open-file").unwrap();
                }
                "save_file" => {
                    event.window().emit("menu-event", "save-file").unwrap();
                }
                _ => {}
            }
        })
        .setup(|app| {
            // Custom setup logic
            let window = app.get_window("main").unwrap();
            
            #[cfg(debug_assertions)]
            {
                window.open_devtools();
            }
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            read_brick_file,
            write_brick_file,
            list_directory,
            create_directory,
            delete_item,
            compile_isa,
            get_system_info
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
