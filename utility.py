import datetime as dt
import CONSTANTS

WD_OFFSETS_PREV = (3, 1, 1, 1, 1, 1, 2)
WD_OFFSETS_NEXT = (1, 1, 1, 1, 3, 2, 1)
def prev_weekday(date):
    return date - dt.timedelta(days=WD_OFFSETS_PREV[date.weekday()])

def next_weekday(date):
    return date + dt.timedelta(days=WD_OFFSETS_NEXT[date.weekday()])

def parseTime(timeStr):
    # Date Format #YYYY-mm-DD = 10 chars
    if len(timeStr) > 10:
        return dt.datetime.strptime(timeStr, CONSTANTS.timeFormat)
    else:
        return dt.datetime.strptime(timeStr, CONSTANTS.dateFormat)