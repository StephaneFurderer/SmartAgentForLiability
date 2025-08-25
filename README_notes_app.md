# Notes Reviewer Streamlit App

A minimal Streamlit application for reviewing claim notes in a timeline format.

## Features

- **Claim Selection**: Dropdown to select from available claim numbers
- **Timeline View**: Notes displayed chronologically like a thread when scrolling down
- **Data Summary**: Sidebar showing overall statistics about the notes
- **Search Functionality**: Search through all notes for specific terms
- **Responsive Layout**: Clean, minimal interface optimized for note review

## How to Run

### Option 1: Run directly with Streamlit
```bash
streamlit run notes_reviewer_app.py
```

### Option 2: Run in background (headless mode)
```bash
streamlit run notes_reviewer_app.py --server.headless true --server.port 8501
```

The app will automatically:
- Create dummy notes data if none exists
- Load and process the notes
- Display the interactive interface

## Usage

1. **Select a Claim**: Use the dropdown to choose a claim number
2. **View Timeline**: Scroll down to see notes in chronological order
3. **Note Details**: Each note shows:
   - Date and time
   - Days since first note
   - Note content
   - Metadata (length, word count, note number)
4. **Sidebar Info**: View overall statistics and search functionality

## Data Structure

The app expects notes data with these columns:
- `note`: The note text content
- `clmNum`: Claim number identifier
- `whenadded`: Timestamp when the note was added

## Dependencies

- streamlit
- pandas
- numpy
- notes_reviewer_agent (custom module)

## File Structure

```
LIGHTHOUSE/
├── notes_reviewer_agent.py    # Core notes processing logic
├── notes_reviewer_app.py      # Streamlit application
├── _data/                     # Data directory (ignored by git)
│   ├── notes.csv             # Notes data
│   └── notes.parquet         # Optimized data format
└── README_notes_app.md       # This file
```

## Customization

The app is designed to be minimal but can be easily extended:
- Add filters for date ranges
- Include note categories or tags
- Add export functionality
- Customize the timeline appearance
- Add user authentication if needed
