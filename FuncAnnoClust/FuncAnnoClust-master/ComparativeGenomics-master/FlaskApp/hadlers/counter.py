import pandas as pd
import polars as pl
import pyarrow
import numpy as np


def countFunctions(data, categories_systems):
    files = data.getClassified()
    strains = list(files.keys())

    # Generate category and system names for the output DataFrame
    categories_names = ["C_" + key for key in categories_systems if categories_systems[key]['selected']]
    systems_names = [
        "S_" + key + "_" + sys
        for key in categories_systems
        for sys in categories_systems[key]["systems"]
    ]

    # Initialize the count DataFrame with zeros
    count_data = {
        "Strain": strains,
        **{name: [0] * len(strains) for name in categories_names + systems_names}
    }
    count = pl.DataFrame(count_data)

    # Count occurrences for each strain
    for strain_name in strains:
        strain_data = files[strain_name]

        # Update counts for categories
        for category_name in categories_names:
            category_key = category_name.replace("C_", "")
            count = count.with_columns(
                pl.when(pl.col("Strain") == strain_name)
                .then(
                    strain_data.filter(
                        pl.col("Category").str.contains(category_key, literal=True)
                    ).height
                )
                .otherwise(pl.col(category_name))
                .alias(category_name)
            )

        # Update counts for systems
        for system_name in systems_names:
            parts = system_name.replace("S_", "").split("_")
            category_key, system_key = parts[0], "_".join(parts[1:])

            # Check if this system is a "no subcategory" entry
            is_no_subcategory = "no subcategory" in system_key

            # Filter the data based on whether it's a "no subcategory" entry or not
            if is_no_subcategory:
                filtered_data = strain_data.filter(
                    (pl.col("Category").str.contains(category_key, literal=True)) &
                    (pl.col("System").str.contains(system_key, literal=True))
                )
            else:
                filtered_data = strain_data.filter(
                    (pl.col("Category").str.contains(category_key, literal=True)) &
                    (pl.col("System").str.contains(system_key, literal=True)) &
                    (~pl.col("System").str.contains("no subcategory", literal=True))
                )

            # Update the count for the system
            count = count.with_columns(
                pl.when(pl.col("Strain") == strain_name)
                .then(filtered_data.height)
                .otherwise(pl.col(system_name))
                .alias(system_name)
            )

    # Sort by 'Strain' and return the result
    return count.sort("Strain")