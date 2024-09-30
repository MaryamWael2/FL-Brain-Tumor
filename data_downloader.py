import logging 
import subprocess
import sys
import os
from pathlib import PurePath

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

USERNAME = "ENTER YOUR EMAIL"
PASSWORD = "ENTER YOUR PASSWORD"
SHAREPOINT_SITE = "ENTER LINK TO SHAREPOINT NAME
SHAREPOINT_SITE_NAME = "ENTER SHARE POINT SITE NAME"
SHAREPOINT_DOC = "ENTER FOLDER NAME"

def save_file(file_n, file_obj):
    file_dir_path = PurePath(".", file_n)
    with open(file_dir_path, 'wb') as f:
        f.write(file_obj)
    
def download_data(file_name, output_folder, folder_name="Data"):
    install("office365-rest-client")
    from office365.sharepoint.client_context import ClientContext
    from office365.runtime.auth.user_credential import UserCredential
    from office365.sharepoint.files.file import File
    logging.error("downloader - package installed")
    
    conn = ClientContext(SHAREPOINT_SITE).with_credentials(
        UserCredential(
            USERNAME,
            PASSWORD
        )
    )
    file_url = f'/sites/{SHAREPOINT_SITE_NAME}/{SHAREPOINT_DOC}/{folder_name}/{file_name}'
    file = File.open_binary(conn, file_url)
    file_obj = file.content
    save_file(file_name, file_obj)
    logging.error("downloader - data downloaded")
    
    install("patool")
    import patoolib  
    patoolib.extract_archive(file_name, outdir=output_folder)
    logging.error("downloader - data unzipped")