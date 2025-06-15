import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV

# Load the order book data
df = pd.read_csv("C:/Users/boshi/Desktop/hw/first_25000_rows.csv")

# Ensure correct ordering
df.sort_values(by="ts_event", inplace=True)
df.reset_index(drop=True, inplace=True)

def compute_best_level_ofi(df):
    """
    Compute Best-Level OFI using Level 0 bid/ask price and size.
    This captures imbalance between buy and sell orders at the top of the book.
    """
    df_prev = df.shift(1)

    of_bid = np.where(
        df['bid_px_00'] > df_prev['bid_px_00'],
        df['bid_sz_00'],
        np.where(df['bid_px_00'] == df_prev['bid_px_00'],
                 df['bid_sz_00'] - df_prev['bid_sz_00'],
                 -df_prev['bid_sz_00'])
    )

    of_ask = np.where(
        df['ask_px_00'] > df_prev['ask_px_00'],
        -df['ask_sz_00'],
        np.where(df['ask_px_00'] == df_prev['ask_px_00'],
                 df['ask_sz_00'] - df_prev['ask_sz_00'],
                 df_prev['ask_sz_00'])
    )

    return pd.Series(of_bid - of_ask, index=df.index, name='best_level_ofi')

df['best_level_ofi'] = compute_best_level_ofi(df)

def compute_multi_level_ofi(df, levels=10):
    """
    Compute multi-level OFI across bid/ask levels 0 to (levels-1).
    """
    ofi_columns = []
    for level in range(levels):
        bid_px = f'bid_px_{level:02d}'
        bid_sz = f'bid_sz_{level:02d}'
        ask_px = f'ask_px_{level:02d}'
        ask_sz = f'ask_sz_{level:02d}'

        df_prev = df.shift(1)

        of_bid = np.where(
            df[bid_px] > df_prev[bid_px],
            df[bid_sz],
            np.where(df[bid_px] == df_prev[bid_px],
                     df[bid_sz] - df_prev[bid_sz],
                     -df_prev[bid_sz])
        )

        of_ask = np.where(
            df[ask_px] > df_prev[ask_px],
            -df[ask_sz],
            np.where(df[ask_px] == df_prev[ask_px],
                     df[ask_sz] - df_prev[ask_sz],
                     df_prev[ask_sz])
        )

        col_name = f'ofi_{level:02d}'
        df[col_name] = of_bid - of_ask
        ofi_columns.append(col_name)

    return df[ofi_columns]

multi_ofi_df = compute_multi_level_ofi(df, levels=10)

def compute_integrated_ofi(df, ofi_columns):
    """
    Perform PCA on multi-level OFIs and extract the first component,
    normalized by L1 norm to form the Integrated OFI.
    """
    ofi_matrix = df[ofi_columns].fillna(0).values
    pca = PCA(n_components=1)
    pc = pca.fit_transform(ofi_matrix).flatten()

    weights = pca.components_[0]
    weights /= np.sum(np.abs(weights))  # L1 normalization

    return pd.Series(ofi_matrix @ weights, index=df.index, name='integrated_ofi')

df['integrated_ofi'] = compute_integrated_ofi(df, [f'ofi_{i:02d}' for i in range(10)])

def compute_cross_asset_ofi(df, symbol_column='symbol', ofi_column='integrated_ofi', target_return_minutes=1):
    """
    Compute Cross-Asset OFI using LASSO regression.
    Predict future return of each symbol using other symbols' current OFIs.
    Returns predicted return as a cross_asset_ofi feature.
    """
    result_df = pd.DataFrame()
    df = df.copy()
    df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2
    df['future_mid_price'] = df.groupby(symbol_column)['mid_price'].shift(-target_return_minutes)
    df['return'] = np.log(df['future_mid_price'] / df['mid_price'])
    df.dropna(subset=[ofi_column, 'return'], inplace=True)

    unique_symbols = df[symbol_column].unique()
    if len(unique_symbols) <= 1:
        print("Cross-Asset OFI skipped: only one symbol in dataset.")
        df['cross_asset_ofi'] = np.nan
        return df[['ts_event', symbol_column, 'cross_asset_ofi']]

    for symbol in unique_symbols:
        sub_df = df[df[symbol_column] == symbol]
        X_full = []
        for other_symbol in unique_symbols:
            if other_symbol == symbol:
                continue
            x = df[df[symbol_column] == other_symbol][ofi_column].values[:len(sub_df)]
            X_full.append(x)
        if len(X_full) == 0:
            continue
        X = np.vstack(X_full).T
        y = sub_df['return'].values

        lasso = LassoCV(cv=5, random_state=0).fit(X, y)
        y_pred = lasso.predict(X)

        temp = sub_df[['ts_event']].copy()
        temp[symbol_column] = symbol
        temp['cross_asset_ofi'] = y_pred
        result_df = pd.concat([result_df, temp], axis=0)

    return result_df

# Only run cross-asset OFI if multiple symbols exist
if df['symbol'].nunique() > 1:
    cross_ofi_df = compute_cross_asset_ofi(df)
    df = pd.merge(df, cross_ofi_df, on=['ts_event', 'symbol'], how='left')
else:
    df['cross_asset_ofi'] = np.nan  # placeholder if only one symbol

df.to_csv("full_ofi_features.csv", index=False)
