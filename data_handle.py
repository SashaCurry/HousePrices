import pandas as pd


# Заполняет null-значения
def __handle_null_values(df: pd.DataFrame) -> pd.DataFrame:
    lotfrontage_null = df.groupby('LotConfig')['LotFrontage'].mean().round()
    for value, mean in lotfrontage_null.items():
        df.loc[(df['LotFrontage'].isnull()) & (df['LotConfig'] == value), 'LotFrontage'] = mean

    df.loc[df['Alley'].isnull(), 'Alley'] = 'Absent'

    df.loc[df['MasVnrType'].isnull(), 'MasVnrType'] = 'Absent'
    df.loc[df['MasVnrArea'].isnull(), 'MasVnrArea'] = 0

    df.loc[df['BsmtQual'].isnull(), 'BsmtQual'] = 'Absent'
    df.loc[df['BsmtCond'].isnull(), 'BsmtCond'] = 'Absent'
    df.loc[df['BsmtExposure'].isnull(), 'BsmtExposure'] = 'Absent'
    df.loc[df['BsmtFinType1'].isnull(), 'BsmtFinType1'] = 'Absent'
    df.loc[(df['BsmtFinType2'].isnull()) & (df['BsmtFinSF2'] > 0), 'BsmtFinType2'] = 'Rec'
    df.loc[df['BsmtFinType2'].isnull(), 'BsmtFinType2'] = 'Absent'

    df.loc[df['Electrical'].isnull(), 'Electrical'] = 'SBrkr'

    df.loc[df['FireplaceQu'].isnull(), 'FireplaceQu'] = 'Absent'

    df.loc[df['GarageType'].isnull(), 'GarageType'] = 'Absent'
    df.loc[df['GarageFinish'].isnull(), 'GarageFinish'] = 'Absent'
    df.loc[df['GarageQual'].isnull(), 'GarageQual'] = 'Absent'
    df.loc[df['GarageCond'].isnull(), 'GarageCond'] = 'Absent'

    df.loc[df['PoolQC'].isnull(), 'PoolQC'] = 'Absent'

    df.loc[df['Fence'].isnull(), 'Fence'] = 'Absent'

    return df


# Фича-инженеринг, обработка категориальных фич
def __handle_features(df: pd.DataFrame, handle_categorical: str = 'None') -> pd.DataFrame:
    pass


# Удаляет ненужные фичи (остаётся 31 значимая)
def __delete_unnecessary_features(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_save = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'Neighborhood_Class', 'TotalBaths_Target',
                       'OverallCond', 'RemodAge', 'LotArea', 'GarageArea', 'BsmtFinSF_Ratio', 'MSSubClass_Rating',
                       'HouseAge', 'KitchenQual_Rating', 'GarageCars_Rating', 'FireplaceQu_Target', 'BsmtQual_Rating',
                       'BsmtExplosure_Rating', 'GarageType_Target', 'TotRmsAbvGrd_Rating', 'ExterQual_Rating',
                       'WoodDeckSF', 'LotFrontage', 'GarageAge' ,'BsmtFinType_Rating', 'SaleCondition_Target',
                       'Fareplaces_Rating', 'Exterior1st_Rating', 'CentralAir', 'LotConfig_Rating',
                       'Condition1_Rating', 'IsPerfectFunctional']

    df = df[columns_to_save]

    return df


def preprocessing(df: pd.DataFrame, handle_categorical: str = 'None') -> pd.DataFrame:

    df = __handle_null_values(df)

    df = __handle_features(df, handle_categorical)

    df = __delete_unnecessary_features(df)

    return df