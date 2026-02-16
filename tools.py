from utility import *

class FuzzyDateSchema(BaseModel):
    ''' Schema for parsing fuzzy dates into a standard date format.
    '''
    how_many_days_from_today: int = Field(0, description="How many days from today. Examples: for 'last year', return '-365';" \
    "for 'last month', return '-30';" \
    "for 'last two weeks', return '-14';" \
    "for 'last week', return '-7';" \
    "for 'yesterday', return '-1';" \
    "for 'today', return '0';" \
    "for 'tomorrow', return '1';" \
    "for 'next week', return '7';" \
    "for 'next two weeks', return '14';" \
    "for 'next month', return '30';" \
    "for 'next year', return '365'.")

class DateSchema(BaseModel):
    ''' Schema for parsing exact dates into a standard date format.
    '''
    date_str: str = Field(..., description="The exact date string to be parsed into 'YYYY-MM-DD' format. Examples: '2023-10-05', 'March 15, 2022', '12/31/2021'.")

@tool(args_schema=FuzzyDateSchema)
async def parse_fuzzy_date(how_many_days_from_today: int) -> str:
    '''
        Returns the number of days between today and a given target date. The result is a signed integer: positive if the target is in the future, zero if it is today, negative if it is in the past. The current date is automatically determined by the system. into a standard date format 'YYYY-MM-DD'.
    '''

    ## How many days from today. into a standard date format 'YYYY-MM-DD'.
    try:
        parse_date = date.fromordinal(date.today().toordinal()+int(how_many_days_from_today)).isoformat()
        print("\n\n------------- parse_fuzzy_date ---------------")
        print(parse_date)
        print("------------- parse_fuzzy_date ---------------\n\n")
    except Exception as e:
        return str(e)
    return parse_date


@tool(args_schema=DateSchema)
async def today(date_str: str) -> str:
    '''
        Returns today into a standard date format 'YYYY-MM-DD'.
    '''

    ## How many days from today. into a standard date format 'YYYY-MM-DD'.
    try:
        parse_date = date.fromordinal(date.today().toordinal()).isoformat()
        print("\n\n------------- today ---------------")
        print(parse_date)
        print("------------- today ---------------\n\n")
    except Exception as e:
        return str(e)
    return parse_date