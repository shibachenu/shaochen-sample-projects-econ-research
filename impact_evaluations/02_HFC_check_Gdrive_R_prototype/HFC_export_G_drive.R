#Automatically export excel spreadsheet to Google drive, notice this code cannot be run standalone, it is integrated with the overall PSSN-HFC chec

#example of running this in Stata is
#shell "$rpath" --vanilla  <"$rwd"/HFC_export_G_drive.R --args hfcoutput="$hfcoutput"

hfc_arg = commandArgs(trailingOnly = TRUE)
hfc_output = strsplit(hfc_arg, "=", fixed = TRUE)[[1]][2]

#If googledrive is not installed, install locally, otherwise load the library
if (!require("googledrive")) install.packages("googledrive")
library(googledrive)

if (!require("tryCatchLog")) install.packages("tryCatchLog")
library(tryCatchLog)


#' Authenticate the main Google account which will be used to upload new sheets and manage folders and permission.
#' This method will also automatically handle errors and warnings and display such messages
#' @param main_google_account String: main email account.
#' @param deauth: logic default FALSE: set to true if want to refresh authentication 

gdrive_auth <- function(main_google_account, deauth = FALSE) {
  tryCatch(
    {
      if (deauth)
        drive_deauth()
      drive_auth(email = main_google_account)
      message("Authentication successful for gmail account: ", main_google_account)
    },
    error = function(e) {
      # Handle error if authentication fails
      message("An error occurred during authentication: ", e$message)
      message("Please try to re-run the script by setting deauth as true")
      # You can add additional handling here, such as re-attempting authentication or logging the error
    },
    warning = function(w) {
      # Handle warnings
      message("A warning occurred: ", w$message)
    },
    finally = {
      message("Authentication attempt finished.")
    }
  )
}


#' Share Google sheets with shared emails passed in, both main summary HFC sheet and flagged sheets. This can be also achieved by going to Google drive webpage. 
#' @param shared_emails: a vectors of emails to share the sheets with, they will have writer permission
#' @param pssn_hfcs: hfc Gsheet file
#' @param pssn_flagged: hfc flagged file
#'
gsheets_share <- function(shared_emails, pssn_hfcs, pssn_flagged) {
  for (shared_email in shared_emails) {
    pssn_hfcs %>%
      drive_share(
        role = "writer",
        type = "user",
        emailAddress = shared_email,
        emailMessage = "PSSN 2 HFC results spreadsheet for collaboration"
      )
    
    pssn_flagged %>%
      drive_share(
        role = "writer",
        type = "user",
        emailAddress = shared_email,
        emailMessage = "PSSN 2 HFC Flagged cases spreadsheet for collaboration"
      )
  }
}


#' Create a new folder with given folder name in G-drive root if not exists, otherwise return existing folder with same name 
#' @param folder_name : string of folder name 
#' @param  folder_path : path of the folder
#' @return path of the folder
#' 
gfolder_put <- function(folder_name, folder_path) {
  #Check if there exists this folder already
  folder = drive_get(path = paste(folder_path, folder_name, "", sep = "/"))
  if(nrow(folder)==0){
    drive_mkdir(name = folder_name, path = folder_path, overwrite = FALSE)
    folder = drive_get(path = paste(folder_path, folder_name, "", sep = "/"))
  }
  return(folder$path)
}

# Step 1: Authenticate Google Account
gdrive_auth(main_google_account = "pssnwb@gmail.com")

# Step 2: Setup folders for renewed uploads of HFC results
endline_folder_path = gfolder_put(folder_name= "PSSN2 Endline", folder_path = "~")
endline_flagged_folder_path = gfolder_put(folder_name= "Flagged_Cases", folder_path = "~/PSSN2 Endline")


# Step 3: Upload/Replace HFC results to Google sheets 
pssn2_endline_hfc_filename = "PSSN2_Endline_HFCs.xlsx"
hfc_output_file =  file.path(hfc_output, pssn2_endline_hfc_filename)
pssn2_endline_flagged_cases_filename = "PSSN2_Endline_Flagged_Cases.xlsx"
hfc_output_flagged =  file.path(hfc_output, pssn2_endline_flagged_cases_filename)

pssn_hfcs = drive_put(hfc_output_file, path = endline_folder_path, name = pssn2_endline_hfc_filename)
pssn_flagged = drive_put(hfc_output_flagged, path = endline_flagged_folder_path, name = pssn2_endline_flagged_cases_filename)

#Step 4: Permission control: share these files with writer permission with users needed
gsheets_share(shared_emails = c("daisyc.reboul@gmail.com", "ifwonderland@gmail.com"), pssn_hfcs, pssn_flagged)

#Step 5: Download this file back to the local folder
drive_download(pssn_flagged, path = file.path(hfc_output, "Downloaded_Endline_Flagged.xlsx"), overwrite = TRUE)