# Fantasy Cricket Project
This project helps you gather and analyze player-wise cricket stats for fantasy cricket applications.

## 1. Get Player-wise Stats

Use the following URL format to fetch player statistics:
```
https://stats.espncricinfo.com/ci/engine/player/PLAYER_ID.html?class=3;opposition=OPPOSITION_ID;ground=GROUND_ID;template=results;type=batting
```
- Replace `PLAYER_ID`, `OPPOSITION_ID`, and `GROUND_ID` with the appropriate values.

## 2. Extract Player, Opposition, and Venue IDs

- Use `run2.py` to obtain the `PLAYER_ID`.
- Find `OPPOSITION_ID` and `GROUND_ID` manually from the ESPN Cricinfo website.

## 3. Python Structure for Jupyter Notebook

- Organize your Jupyter notebook with the following structure:
    1. **Import Libraries**  
        Import required Python libraries (e.g., `requests`, `pandas`, `BeautifulSoup`).
    2. **Define Helper Functions**  
        Functions for fetching and parsing player stats.
    3. **Input Section**  
        Input player, opposition, and venue IDs.
    4. **Data Extraction**  
        Use the helper functions to fetch and process data.
    5. **Analysis & Visualization**  
        Analyze and visualize the extracted stats.
    6. **Conclusion**  
        Summarize findings or next steps.

## 4. End-to-End Connection
- Ensure all steps are connected: from fetching IDs, extracting data, to analysis and visualization, for a seamless workflow in your Jupyter notebook.
