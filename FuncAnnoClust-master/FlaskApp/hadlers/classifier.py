import polars as pl
from hadlers.keywordsClassifier import keywordsClassify
from hadlers.validator import validate


def _clean_string(value):
    if isinstance(value, str):
        value = value.replace("&#39;", "'")
        value = value.replace("<br>", "")
        value = value.replace("'", "")
        value = value.strip()
    return value


def _last_from_semicolon(value):
    if isinstance(value, str):
        return value.split(";")[-1].strip()
    return value


def _as_str(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _is_empty_df(df):
    return df is None or len(df) == 0


def _get_user_cls(data):
    if hasattr(data, "getUserCls"):
        return data.getUserCls()
    return getattr(data, "userCls", None)


def _get_kw_cls(data):
    if hasattr(data, "getKwCls"):
        return data.getKwCls()
    return getattr(data, "kwCls", None)


def _build_lookup(df, key_col, value_cols):
    lookup = {}

    if _is_empty_df(df):
        return lookup

    needed_cols = [key_col] + list(value_cols)

    if any(col not in df.columns for col in needed_cols):
        return lookup

    sub = df.select(needed_cols)

    keys = sub[key_col].to_list()
    values = sub.select(value_cols).rows()

    for key, value_row in zip(keys, values):
        if key is None:
            continue

        if not isinstance(key, str):
            continue

        if key not in lookup:
            lookup[key] = []

        lookup[key].append(value_row)

    return lookup


def _add_rank(current, value):
    if value is None:
        return current

    value = str(value).strip()

    if not value:
        return current

    if current is None:
        current_str = "none"
    elif isinstance(current, str):
        current_str = current
    else:
        current_str = str(current)

    if "none" in current_str and len(current_str.split("; <br>")) <= 1:
        return value

    if current_str:
        parts = [p.strip() for p in current_str.split(";")]

        if value not in parts:
            parts.append(value)
            return "; ".join(sorted(parts))

    return current


def _match_row(match_type, lookup, row_vals):
    source = row_vals.get(match_type)
    source = _as_str(source)

    if not source:
        return 0

    match_count = 0

    for raw_value in source.split("; <br>"):
        key = raw_value.strip()
        matched_rows = lookup.get(key)

        if not matched_rows:
            continue

        for matched_row in matched_rows:
            if len(matched_row) > 0:
                row_vals["Category"] = _add_rank(
                    row_vals["Category"],
                    matched_row[0]
                )

            if len(matched_row) > 1:
                row_vals["System"] = _add_rank(
                    row_vals["System"],
                    matched_row[1]
                )

            match_count += 1

            if match_type == "Function" and len(matched_row) > 2:
                row_vals["Subsystem"] = _add_rank(
                    row_vals["Subsystem"],
                    matched_row[2]
                )

    return match_count


def _legacy_classify_flag(row_flags, mode_used):
    if mode_used is None:
        return False

    if mode_used == 2:
        return None if row_flags else False

    return row_flags[-1] if row_flags else False


def classifyFunctions(
    cls_types,
    files,
    data,
    *,
    strict_legacy_classified=False
):
    resultsList = data.getClassified()
    displayError = ""

    output_schema = {
        "Category": pl.Utf8,
        "System": pl.Utf8,
        "Subsystem": pl.Utf8,
        "Function": pl.Utf8,
        "Is_None": pl.Boolean,
    }

    for file in files:
        error, fileContent = validate(file, "rastDownload")

        if len(error) > 0:
            print(f"Error processing file {file.filename}: {error}")
            displayError = error
            continue

        if not {"System", "Category"}.issubset(fileContent.columns):
            fileContent = fileContent.with_columns([
                pl.lit("none").alias("System"),
                pl.lit("none").alias("Category"),
            ])
            fileContent = fileContent.select([
                "Category",
                "System",
                "Subsystem",
                "Function"
            ])

        fileContent_none = fileContent.filter(pl.col("Subsystem") == "- none -")
        fileContent = fileContent.filter(pl.col("Subsystem") != "- none -")

        fileContent_none = fileContent_none.with_columns(
            pl.lit(True).alias("Is_None")
        )
        fileContent = fileContent.with_columns(
            pl.lit(False).alias("Is_None")
        )

        base_rows = fileContent.select([
            "Category",
            "System",
            "Subsystem",
            "Function"
        ]).rows()

        out_rows = []
        row_flags = []
        mode_used = None

        classified = False

        user_cls = None
        user_cls_exists = False

        if "0" not in cls_types and "1" in cls_types:
            user_cls = _get_user_cls(data)
            user_cls_exists = not _is_empty_df(user_cls)

        if "0" in cls_types:
            mode_used = 0

            if base_rows:
                rast_lookup = _build_lookup(
                    data.getRastCls(),
                    "Subsystem",
                    ["Category", "System"]
                )
            else:
                rast_lookup = {}

            for cat, sys, sub, func in base_rows:
                row_vals = {
                    "Category": cat,
                    "System": sys,
                    "Subsystem": sub,
                    "Function": func,
                }

                match_count = _match_row(
                    "Subsystem",
                    rast_lookup,
                    row_vals
                )

                row_flags.append(match_count > 0)

                cat_clean = _clean_string(row_vals["Category"])
                sys_clean = _clean_string(row_vals["System"])
                sub_clean = _clean_string(row_vals["Subsystem"])

                out_rows.append((
                    _last_from_semicolon(cat_clean),
                    _last_from_semicolon(sys_clean),
                    _last_from_semicolon(sub_clean),
                    func,
                    False,
                ))

        elif "1" in cls_types and not classified and user_cls_exists:
            mode_used = 1

            if base_rows:
                user_lookup = _build_lookup(
                    user_cls,
                    "Function",
                    ["Category", "System", "Subsystem"]
                )
            else:
                user_lookup = {}

            for cat, sys, sub, func in base_rows:
                row_vals = {
                    "Category": cat,
                    "System": sys,
                    "Subsystem": sub,
                    "Function": func,
                }

                match_count = _match_row(
                    "Function",
                    user_lookup,
                    row_vals
                )

                row_flags.append(match_count > 0)

                out_rows.append((
                    _last_from_semicolon(row_vals["Category"]),
                    _last_from_semicolon(row_vals["System"]),
                    _last_from_semicolon(row_vals["Subsystem"]),
                    func,
                    False,
                ))

        elif "2" in cls_types and not classified:
            mode_used = 2

            kw_cache = {
                "height": None,
                "lookup": {},
            }

            def get_kw_lookup():
                kw_df = _get_kw_cls(data)

                if _is_empty_df(kw_df):
                    if kw_cache["height"] != 0:
                        kw_cache["height"] = 0
                        kw_cache["lookup"] = {}
                    return kw_cache["lookup"]

                current_height = kw_df.height

                if kw_cache["height"] != current_height:
                    kw_cache["lookup"] = _build_lookup(
                        kw_df,
                        "Function",
                        ["Category", "System", "Subsystem"]
                    )
                    kw_cache["height"] = current_height

                return kw_cache["lookup"]

            for cat, sys, sub, func in base_rows:
                row_vals = {
                    "Category": cat,
                    "System": sys,
                    "Subsystem": sub,
                    "Function": func,
                }

                matched_any = False
                func_str = _as_str(func)

                for value in func_str.split("; <br>"):
                    keywordsClassify(value, data)

                    match_count = _match_row(
                        "Function",
                        get_kw_lookup(),
                        row_vals
                    )

                    if match_count > 0:
                        matched_any = True

                row_flags.append(matched_any)

                out_rows.append((
                    _last_from_semicolon(row_vals["Category"]),
                    _last_from_semicolon(row_vals["System"]),
                    _last_from_semicolon(row_vals["Subsystem"]),
                    func,
                    False,
                ))

        else:
            for cat, sys, sub, func in base_rows:
                out_rows.append((
                    _last_from_semicolon(cat),
                    _last_from_semicolon(sys),
                    _last_from_semicolon(sub),
                    func,
                    False,
                ))

        if out_rows:
            fileContent = pl.DataFrame(
                out_rows,
                schema=output_schema,
                orient="row"
            )
        else:
            fileContent = pl.DataFrame(schema=output_schema)

        if strict_legacy_classified:
            classified = _legacy_classify_flag(row_flags, mode_used)
        else:
            classified = any(row_flags) if mode_used is not None else False

        try:
            grouped = fileContent.group_by(
                ["Category", "Function"],
                maintain_order=True
            )
        except TypeError:
            grouped = fileContent.group_by(
                ["Category", "Function"]
            )

        fileContent = grouped.agg(
            pl.first("System").alias("System"),
            pl.first("Subsystem").alias("Subsystem"),
            pl.first("Is_None").alias("Is_None"),
        )

        fileContent = fileContent.sort(by=["Category"]).with_columns(
            pl.when(pl.col("Category") == "Clustering-based subsystems")
            .then(1)
            .when(pl.col("Category") == "none")
            .then(2)
            .otherwise(0)
            .alias("Priority")
        )

        fileContent = fileContent.sort(by=["Function", "Priority"]).unique(
            subset=["Function"],
            keep="first"
        )

        fileContent = fileContent.drop(["Priority", "Is_None"])

        fileContent_none = fileContent_none.select(fileContent.columns)
        fileContent = pl.concat([fileContent, fileContent_none])

        if classified:
            filename = ".".join(file.filename.split(".")[:-1])
            resultsList[filename] = fileContent

    return displayError, resultsList
