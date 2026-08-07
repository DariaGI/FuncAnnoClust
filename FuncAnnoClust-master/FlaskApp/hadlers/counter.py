import polars as pl


def countFunctions(data, categories_systems):
    files = data.getClassified()
    strains = list(files.keys())

    categories_names = ["C_" + key for key in categories_systems if categories_systems[key]['selected']]
    systems_names = [
        "S_" + key + "_" + sys
        for key in categories_systems
        for sys in categories_systems[key]["systems"]
    ]

    parsed_categories = [(name, name[2:]) for name in categories_names]

    parsed_systems = []
    for system_name in systems_names:
        raw = system_name[2:]
        parts = raw.split("_", 1)
        category_key = parts[0]
        system_key = parts[1] if len(parts) > 1 else ""
        is_no_subcategory = "no subcategory" in system_key
        parsed_systems.append((system_name, category_key, system_key, is_no_subcategory))

    result_rows = []

    for strain_name in strains:
        strain_data = files[strain_name]
        row = {"Strain": strain_name}

        if strain_data.height == 0:
            for col_name, _ in parsed_categories:
                row[col_name] = 0
            for col_name, _, _, _ in parsed_systems:
                row[col_name] = 0
            result_rows.append(row)
            continue

        cat_counts = (
            strain_data
            .group_by("Category")
            .agg(pl.len().alias("cnt"))
        )
        cat_values = cat_counts["Category"].to_list()
        cat_cnts = cat_counts["cnt"].to_list()

        for col_name, cat_key in parsed_categories:
            row[col_name] = sum(
                cc for cv, cc in zip(cat_values, cat_cnts)
                if cv is not None and cat_key in cv
            )

        sys_counts = (
            strain_data
            .group_by(["Category", "System"])
            .agg(pl.len().alias("cnt"))
        )
        sys_cats = sys_counts["Category"].to_list()
        sys_vals = sys_counts["System"].to_list()
        sys_cnts = sys_counts["cnt"].to_list()

        for col_name, cat_key, sys_key, is_no_sub in parsed_systems:
            total = 0
            for sc, sv, sn in zip(sys_cats, sys_vals, sys_cnts):
                if sc is None or sv is None:
                    continue
                if cat_key in sc and sys_key in sv:
                    if is_no_sub:
                        total += sn
                    elif "no subcategory" not in sv:
                        total += sn
            row[col_name] = total

        result_rows.append(row)

    schema = {"Strain": pl.Utf8}
    for name in categories_names + systems_names:
        schema[name] = pl.UInt32

    count = pl.DataFrame(result_rows, schema=schema)
    return count.sort("Strain")
