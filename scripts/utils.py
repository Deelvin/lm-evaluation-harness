import gspread

TABLE_ID = "1DQEo8_6uj1ZnpIuk69jX9LFc1HUXg_4gElBN9eHjeUU"

def init_gspread_client() -> gspread.spreadsheet.Spreadsheet:
    gc = gspread.oauth()
    return gc.open_by_key(TABLE_ID)