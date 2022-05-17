"""Trim df."""
import pandas as pd


# fmt: off
def trim_df(
        df1: pd.DataFrame,
        len_: int = 4,
) -> pd.DataFrame:
    # fmt: on
    """Trim df."""
    if len(df1) > 2 * len_:
        df_trimmed = pd.concat(
            [
                df1.iloc[:len_, :],
                pd.DataFrame(
                    # [["...", "...",]],
                    [["..."] * len(df1.columns)],
                    columns=df1.columns,
                ),
                df1.iloc[-len_:, :],
            ],
            ignore_index=1,
        )
        return df_trimmed
    return df1
