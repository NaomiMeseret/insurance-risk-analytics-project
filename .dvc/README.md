# DVC Configuration Directory

This directory contains DVC (Data Version Control) configuration files.

## Files

- **`config`**: DVC remote storage configuration

  - Remote name: `localstorage`
  - Remote URL: `~/dvc_storage`
  - Default remote: Yes

- **`.gitignore`**: Files to exclude from Git (DVC cache and temp files)

## Configuration Details

The DVC configuration is set up with:

- **Local remote storage** at `~/dvc_storage`
- **Autostage enabled**: Automatically stages files when using `dvc add`
- **Checksum jobs**: 2 parallel jobs for faster processing

## Usage

This configuration allows you to:

1. Track large data files with DVC
2. Store data in local remote storage
3. Version control data through Git (via .dvc files)
4. Reproduce analyses with exact data versions

## Verification

To verify the configuration:

```bash
dvc remote list
```

Expected output:

```
localstorage	~/dvc_storage
```
