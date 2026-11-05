import numpy as np
import pandas as pd
import re


def parse_file_trial_duration_info(filename):
    """Parse the filename to extract trial label and burst on duration information.

    Args:
        filename (str): The filename to parse.

    Returns:
        dict: A dictionary containing the extracted information, including:
            - filename (str): The original filename.
            - ind_num (int): The individual number extracted from the filename.
            - trial_num (int): The trial number extracted from the filename.
            - freq_num (int): The frequency number extracted from the filename.
            - pulse_width_num (int): The pulse width number extracted from the filename.
            - trial_label (str): A formatted string representing the trial label and burst on duration; if burst duration is missing,
              the label includes frequency and pulse width information.
            - formatted_dur_num (float): The burst on duration extracted from the filename, formatted as a float.
    """

    # Extract trial label and burst on duration (e.g. {'01','1s'}, {'02','2s'}, etc.)
    match = re.search(
        r"ind(?:_)?(\d{2})_(\d+)_(?:.*?(?=\d+hz|\d+ms|\d+s|DLC))?(?:(\d+)hz)?(?:_)?(?:(\d+)ms)?(?:_)?(?:(\d+)s)?DLC",
        filename,
    )

    if not match:
        return {
            "filename": filename,
            "ind_num": 0,
            "trial_num": 0,
            "freq_num": 0,
            "pulse_width_num": 1,
            "trial_label": "0.0",
            "formatted_dur_num": 0.0,
        }

    if match:
        ind_str, trial_str, freq_str, pulse_width, dur_str = match.groups()
        ind_num = int(ind_str)
        trial_num = int(trial_str)
        freq_num = int(freq_str) if freq_str else 0
        pulse_width_num = int(pulse_width) if pulse_width else 1

        if dur_str:
            # Insert decimal point if ID has 3 or more digits ('201' -> '2.1')
            if len(dur_str) >= 3 or dur_str.startswith("0"):
                formatted_dur_str = f"{dur_str[0]}.{dur_str[-1]}"
                formatted_dur_num = float(formatted_dur_str)
            else:
                formatted_dur_num = float(dur_str)
                formatted_dur_str = dur_str

            trial_label = f"Trial {trial_str} - Burst ON {formatted_dur_num}s"
        else:
            formatted_dur_num = 0.5
            trial_label = f"Trial {trial_str} - Burst ON 0.5s, Frequency {freq_num}Hz, Pulse Width {pulse_width_num}ms"

    # Guard against any code path where the regex does not provide Freq/Pulse values
    if "freq_num" not in locals():
        freq_num = 0
    if "pulse_width_num" not in locals():
        pulse_width_num = 1

    return {
        "filename": filename,
        "ind_num": ind_num,
        "trial_num": trial_num,
        "freq_num": freq_num,
        "pulse_width_num": pulse_width_num,
        "trial_label": trial_label,
        "formatted_dur_num": formatted_dur_num,
    }


def filter_dataframe(df, include_likelihood=False):
    """Filter the DataFrame to extract relevant columns and compute angles based on body part coordinates.

    Args:
        df (pd.DataFrame): The input DataFrame containing body part coordinates and likelihoods.
        include_likelihood (bool): Whether to include likelihood columns in the output DataFrame.
    Returns:
        pd.DataFrame: A filtered DataFrame with computed angles and relevant columns.
    Raises:
        ValueError: If the input DataFrame does not have the expected structure (at least
        3 rows and 2 columns, with the first two rows containing body part and coordinate headers).
    """

    # Validate DataFrame structure before extracting body parts and coordinates
    if df.shape[0] < 3 or df.shape[1] < 2:
        raise ValueError("Input DataFrame does not have the expected shape (at least 3 rows and 2 columns).")
    if not isinstance(df.iloc[0, 1], str) or not isinstance(df.iloc[1, 1], str):
        raise ValueError(  # noqa: TRY004
            "First two rows after the first column should contain body part and coordinate headers as strings."
        )

    bodyparts = df.iloc[0, 1:]
    coords = df.iloc[1, 1:]

    df_filtered = df.iloc[2:].reset_index(drop=True)  # remove the first two rows which are not needed

    flat_cols = ["frame"] + [f"{bp}_{c}" for bp, c in zip(bodyparts, coords)]

    df_filtered.columns = flat_cols
    df_filtered = df_filtered.astype({col: float for col in df_filtered.columns if col != "frame"})

    if include_likelihood is False:
        df_filtered.drop(list(df_filtered.filter(regex="likelihood")), axis=1, inplace=True)

    if "FT_Middle_Leg_y" in df_filtered.columns and "FT_Middle_Leg_x" in df_filtered.columns:
        # Calculate the angle between the TCTC and TCFT vectors
        df_filtered["TCTC_vector_x"] = df_filtered["TC_Middle_Leg_x"] - df_filtered["TC_Front_Leg_x"]
        df_filtered["TCTC_vector_y"] = df_filtered["TC_Middle_Leg_y"] - df_filtered["TC_Front_Leg_y"]

        df_filtered["TCFT_vector_x"] = df_filtered["TC_Middle_Leg_x"] - df_filtered["FT_Middle_Leg_x"]
        df_filtered["TCFT_vector_y"] = df_filtered["TC_Middle_Leg_y"] - df_filtered["FT_Middle_Leg_y"]
    else:
        # Calculate the angle between the TCTC and TCFT vectors
        df_filtered["TCTC_vector_x"] = df_filtered["TC_Middle_Leg_x"] - df_filtered["TC_Front_Leg_x"]
        df_filtered["TCTC_vector_y"] = df_filtered["TC_Middle_Leg_y"] - df_filtered["TC_Front_Leg_y"]

        df_filtered["TCFT_vector_x"] = df_filtered["TC_Middle_Leg_x"] - df_filtered["FT_Middle_leg_x"]
        df_filtered["TCFT_vector_y"] = df_filtered["TC_Middle_Leg_y"] - df_filtered["FT_Middle_leg_y"]

    # Dot product and magnitudes of the two vectors to get the cosine of the angle between them
    df_filtered["dot_product"] = (
        df_filtered["TCTC_vector_x"] * df_filtered["TCFT_vector_x"] + df_filtered["TCTC_vector_y"] * df_filtered["TCFT_vector_y"]
    )

    df_filtered["TCTC_magnitude"] = (df_filtered["TCTC_vector_x"] ** 2 + df_filtered["TCTC_vector_y"] ** 2) ** 0.5
    df_filtered["TCFT_magnitude"] = (df_filtered["TCFT_vector_x"] ** 2 + df_filtered["TCFT_vector_y"] ** 2) ** 0.5

    df_filtered["cosine_angle"] = (df_filtered["dot_product"] / (df_filtered["TCTC_magnitude"] * df_filtered["TCFT_magnitude"])).clip(
        -1.0, 1.0
    )

    # Convert frames into seconds
    df_filtered["time"] = pd.to_numeric(df_filtered["frame"], errors="coerce") / 60  # Assuming 60 FPS

    df_filtered["angle"] = np.arccos(df_filtered["cosine_angle"]) * 180 / np.pi

    first_half_sec = df_filtered[df_filtered["time"] <= 0.5]

    # Extract the minimum angle within those first 0.5s (with fallback if missing)
    if not first_half_sec.empty:
        initial_angle = first_half_sec["angle"].min()
    else:
        if not df_filtered["angle"].empty:
            initial_angle = df_filtered["angle"].iloc[0]
        else:
            initial_angle = np.nan

    # Store baseline and compute relative angle shift
    df_filtered["initial_angle"] = initial_angle
    df_filtered["angle_from_initial"] = df_filtered["angle"] - initial_angle

    return df_filtered
