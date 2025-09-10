# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import os
from pathlib import Path

# __file__ ist beim PyInstaller-Spec-Run nicht garantiert.
try:
    BASEDIR = Path(__file__).resolve().parent
except NameError:
    BASEDIR = Path.cwd()

ICON_PATH = BASEDIR / "icon.ico"
VERSION_PATH = BASEDIR / "version.txt"

a = Analysis(
    ['controller_app.py'],
    pathex=[str(BASEDIR)],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Entferne 'calibration_dialog', falls du die Datei nicht hast:
        'calibration_dialog',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Triple One Controller',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # wie --windowed
    icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    version=str(VERSION_PATH) if VERSION_PATH.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Triple One Controller',
)
