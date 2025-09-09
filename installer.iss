#define AppName "Triple One Controller"
#define AppVersion "0.1.0"
#define Publisher "Triple One"
#define InstallDir "{pf}\Triple One Controller"

[Setup]
AppId={{7C5B2A1F-3C1C-4B13-A9AD-1B5C0F5A9A11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
DefaultDirName={#InstallDir}
DefaultGroupName={#AppName}
OutputBaseFilename=Triple-One-Controller-{#AppVersion}-Setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest

[Files]
; Nimm die von PyInstaller erzeugten Dateien
Source: "dist\Triple One Controller\*"; DestDir: "{app}"; Flags: recursesubdirs replacesameversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\Triple One Controller.exe"
Name: "{commondesktop}\{#AppName}"; Filename: "{app}\Triple One Controller.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Desktopverknüpfung erstellen"; GroupDescription: "Zusätzliche Aufgaben:"; Flags: unchecked
