import polars as pl


def all_classified(data):
    classified_files = data.getClassified()
    sorted_bin_names = sorted(classified_files.keys())
    combined_rows = []
    for bin_name, df in classified_files.items():
        # Add a binary column for the current bin
        df_with_bin = df.with_columns(pl.lit(1).alias(bin_name))

        # Add missing binary columns for other bins (with default value 0)
        for other_bin in sorted_bin_names:
            if other_bin != bin_name:
                df_with_bin = df_with_bin.with_columns(pl.lit(0).alias(other_bin))

        # Reorder columns to ensure consistent schema
        df_with_bin = df_with_bin.select(['Category', 'System', 'Subsystem', 'Function'] + sorted_bin_names)

        combined_rows.append(df_with_bin)

    # Step 3: Concatenate all DataFrames into one
    combined_df = pl.concat(combined_rows)

    # Step 4: Group by unique combinations of Category, Function, System, Subsystem
    combined_df = combined_df.group_by(['Category', 'System', 'Subsystem', 'Function']).agg(
        *[pl.max(bin_name).alias(bin_name) for bin_name in sorted_bin_names]
    )
    return combined_df
