import numpy as np
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
def __handle_features(df: pd.DataFrame) -> pd.DataFrame:
    df['MSSubClass_Rating'] = 0
    mssubclass_sorted = df.groupby('MSSubClass', as_index=False)['SalePrice'].mean().sort_values('SalePrice')
    for idx, mssubclass in enumerate(mssubclass_sorted['MSSubClass']):
        df.loc[df['MSSubClass'] == mssubclass, 'MSSubClass_Rating'] = idx + 1
    # df = pd.get_dummies(data=df, columns=['MSSubClass'], drop_first=True)

    df['LotConfig_Rating'] = 0.0
    df.loc[df['LotConfig'] == 'Inside', 'LotConfig_Rating'] = 1
    df.loc[df['LotConfig'].isin(['FR2', 'FR3']), 'LotConfig_Rating'] = 1.25
    df.loc[df['LotConfig'] == 'Corner', 'LotConfig_Rating'] = 1.5
    df.loc[df['LotConfig'] == 'CulDSac', 'LotConfig_Rating'] = 3.5

    df['Neighborhood_Rating'] = 5.9  # Среднее значение по умолчанию
    highest_neighborhood = ['NoRidge', 'NridgHt', 'StoneBr']
    high_neighborhood = ['Timber', 'Veenker', 'Somerst', 'ClearCr', 'Crawfor']
    medium_neighborhood = ['CollgCr', 'Blmngtn', 'NWAmes', 'SawyerW', 'Gilbert', 'Mitchel', 'NPkVill', 'NAmes']
    low_neighborhood = ['Sawyer', 'SWISU', 'Blueste', 'BrkSide', 'Edwards', 'OldTown', 'BrDale']
    lowest_neighborhood = ['IDOTRR', 'MeadowV']
    df.loc[df['Neighborhood'].isin(lowest_neighborhood), 'Neighborhood_Rating'] = 1
    df.loc[df['Neighborhood'].isin(low_neighborhood), 'Neighborhood_Rating'] = 2.5
    df.loc[df['Neighborhood'].isin(medium_neighborhood), 'Neighborhood_Rating'] = 5
    df.loc[df['Neighborhood'].isin(high_neighborhood), 'Neighborhood_Rating'] = 7
    df.loc[df['Neighborhood'].isin(highest_neighborhood), 'Neighborhood_Rating'] = 12

    df['Condition1_Rating'] = 0.0
    good_condition1 = ['PosN', 'PosA', 'RRNn', 'RRNe']
    normal_condition1 = ['Norm', 'RRAn']
    poor_condition1 = ['Feedr', 'Artery', 'RRAe']
    df.loc[df['Condition1'].isin(poor_condition1), 'Condition1_Rating'] = 1
    df.loc[df['Condition1'].isin(normal_condition1), 'Condition1_Rating'] = 4.5
    df.loc[df['Condition1'].isin(good_condition1), 'Condition1_Rating'] = 7

    df['Exterior1st_Rating'] = 0.0
    outdated_exterior1st = ['AsbShng', 'BrkComm', 'AsphShn', 'CBlock', 'BrkComm', 'PreCast', 'Other']
    wooden_exterior1st = ['Wd Sdng', 'Plywood', 'WdShing']
    standart_exterior1st = ['VinylSd', 'MetalSd', 'HdBoard']
    premium_exterior1st = ['BrkFace', 'Stone', 'CemntBd', 'Stucco', 'ImStucc']
    df.loc[df['Exterior1st'].isin(outdated_exterior1st), 'Exterior1st_Rating'] = 1
    df.loc[df['Exterior1st'].isin(wooden_exterior1st), 'Exterior1st_Rating'] = 3.5
    df.loc[df['Exterior1st'].isin(standart_exterior1st), 'Exterior1st_Rating'] = 4
    df.loc[df['Exterior1st'].isin(premium_exterior1st), 'Exterior1st_Rating'] = 5

    df['ExterQual_Rating'] = 1.0  # Сразу объединяю и присваиваю значение для Poor & Fair
    df.loc[df['ExterQual'] == 'TA', 'ExterQual_Rating'] = 2.5
    df.loc[df['ExterQual'] == 'Gd', 'ExterQual_Rating'] = 5
    df.loc[df['ExterQual'] == 'Ex', 'ExterQual_Rating'] = 9

    df['BsmtQual_Rating'] = 1.5  # Значение по умолчанию для Poor
    df.loc[df['BsmtQual'] == 'Absent', 'BsmtQual_Rating'] = 1
    df.loc[df['BsmtQual'] == 'Fa', 'BsmtQual_Rating'] = 1.5
    df.loc[df['BsmtQual'] == 'TA', 'BsmtQual_Rating'] = 3.5
    df.loc[df['BsmtQual'] == 'Gd', 'BsmtQual_Rating'] = 5.5
    df.loc[df['BsmtQual'] == 'Ex', 'BsmtQual_Rating'] = 12.5

    df['BsmtExposure_Rating'] = 0.0
    df.loc[df['BsmtExposure'] == 'Absent', 'BsmtExposure_Rating'] = 1
    df.loc[df['BsmtExposure'] == 'No', 'BsmtExposure_Rating'] = 4
    df.loc[df['BsmtExposure'] == 'Mn', 'BsmtExposure_Rating'] = 5.5
    df.loc[df['BsmtExposure'] == 'Av', 'BsmtExposure_Rating'] = 6
    df.loc[df['BsmtExposure'] == 'Gd', 'BsmtExposure_Rating'] = 8.5

    df['BsmtFinType_Target'] = 0
    lotshape_target = df.groupby('BsmtFinType1')['SalePrice'].mean()
    for bsmtfintype1, target_mean in lotshape_target.items():
        df.loc[df['BsmtFinType1'] == bsmtfintype1, 'BsmtFinType_Target'] = round(target_mean)

    df['BsmtFinSF_Ratio'] = np.where(df['TotalBsmtSF'] > 0,
                                     (df['BsmtFinSF1'] + df['BsmtFinSF2']) / df['TotalBsmtSF'],
                                     0)
    df['BsmtFinSF_Ratio'] = round(df['BsmtFinSF_Ratio'], 1)

    df['CentralAir'] = (df['CentralAir'] == 'Y').astype(int)

    return df


# Удаляет ненужные фичи (остаётся 31 значимая)
def __delete_unnecessary_features(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_save = ['MSSubClass_Rating', 'LotConfig_Rating', 'Neighborhood_Rating', 'Condition1_Rating',
                       'Exterior1st_Rating', 'ExterQual_Rating', 'BsmtQual_Rating', 'BsmtExposure_Rating',
                       'BsmtFinType_Rating', 'BsmtFinSF_Ratio', 'CentralAir']

    old_columns_to_save = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'TotalBaths_Target',
                       'OverallCond', 'RemodAge', 'LotArea', 'GarageArea'
                       'HouseAge', 'KitchenQual_Rating', 'GarageCars_Rating', 'FireplaceQu_Target',
                       'GarageType_Target', 'TotRmsAbvGrd_Rating',
                       'WoodDeckSF', 'LotFrontage', 'GarageAge' , 'SaleCondition_Target',
                       'Fareplaces_Rating',
                       'IsPerfectFunctional']

    df = df[columns_to_save]

    return df


def preprocessing(df: pd.DataFrame, handle_categorical: str = 'None') -> pd.DataFrame:

    df = __handle_null_values(df)

    df = __handle_features(df)

    df = __delete_unnecessary_features(df)

    return df