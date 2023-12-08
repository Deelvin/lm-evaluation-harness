import gspread

TABLE_ID = "13edhIf-wH-mdmzIyvceAwJcujlEDWPvIwcQwr4TdI9U"

def init_gspread_client() -> gspread.spreadsheet.Spreadsheet:
    gc = gspread.oauth()
    return gc.open_by_key(TABLE_ID)