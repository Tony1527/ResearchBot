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

@tool(args_schema=FuzzyDateSchema)
async def parse_fuzzy_date(how_many_days_from_today: int) -> str:
    '''
        How many days from today. into a standard date format 'YYYY-MM-DD'.
    '''
    try:
        parse_date = date.fromordinal(date.today().toordinal()+int(how_many_days_from_today)).isoformat()
        print("------------- parse_fuzzy_date ---------------")
        print(parse_date)
        print("------------- parse_fuzzy_date ---------------")
    except Exception as e:
        return str(e)
    return parse_date