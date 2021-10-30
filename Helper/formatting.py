# method to transform the data from wide to long format
def wide_to_long(data):
    df = company_df_long_format_rrr.stack().reset_index()
    df.columns = ['Date', 'Name', 'RRR']
    df2.columns = ['Date', 'Name', 'R']
    df['Country'] = country
    df = df.merge(df2, on=['Date', 'Name'], how='right')
    df.fillna(value=0)
    return df