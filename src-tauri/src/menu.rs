use tauri::{AboutMetadata, CustomMenuItem, Menu, MenuItem, Submenu};

pub fn get_menu() -> Menu {
    let check_updates = CustomMenuItem::new("check_updates".to_string(), "Check for Updates");
    let preferences = CustomMenuItem::new("preferences".to_string(), "Preferences...");
    
    let app_menu = Submenu::new(
        "BRICK",
        Menu::new()
            .add_native_item(MenuItem::About("BRICK".to_string(), AboutMetadata::default()))
            .add_native_item(MenuItem::Separator)
            .add_item(preferences)
            .add_item(check_updates)
            .add_native_item(MenuItem::Separator)
            .add_native_item(MenuItem::Services)
            .add_native_item(MenuItem::Separator)
            .add_native_item(MenuItem::Hide)
            .add_native_item(MenuItem::HideOthers)
            .add_native_item(MenuItem::ShowAll)
            .add_native_item(MenuItem::Separator)
            .add_native_item(MenuItem::Quit),
    );

    let file_menu = Submenu::new(
        "File",
        Menu::new()
            .add_item(CustomMenuItem::new("new_file", "New File").accelerator("CmdOrCtrl+N"))
            .add_item(CustomMenuItem::new("open_file", "Open...").accelerator("CmdOrCtrl+O"))
            .add_native_item(MenuItem::Separator)
            .add_item(CustomMenuItem::new("save_file", "Save").accelerator("CmdOrCtrl+S"))
            .add_item(CustomMenuItem::new("save_as", "Save As...").accelerator("CmdOrCtrl+Shift+S"))

            .add_native_item(MenuItem::Separator)
            .add_native_item(MenuItem::CloseWindow),
    );

    let edit_menu = Submenu::new(
        "Edit",
        Menu::new()
            .add_native_item(MenuItem::Undo)
            .add_native_item(MenuItem::Redo)
            .add_native_item(MenuItem::Separator)
            .add_native_item(MenuItem::Cut)
            .add_native_item(MenuItem::Copy)
            .add_native_item(MenuItem::Paste)
            .add_native_item(MenuItem::SelectAll),
    );

    let view_menu = Submenu::new(
        "View",
        Menu::new()
            .add_native_item(MenuItem::EnterFullScreen),
    );

    let window_menu = Submenu::new(
        "Window",
        Menu::new()
            .add_native_item(MenuItem::Minimize)
            .add_native_item(MenuItem::Zoom),
    );

    Menu::new()
        .add_submenu(app_menu)
        .add_submenu(file_menu)
        .add_submenu(edit_menu)
        .add_submenu(view_menu)
        .add_submenu(window_menu)
}
