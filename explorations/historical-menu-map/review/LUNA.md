# Independent Luna review

The user explicitly requested an independent Luna review. A separate `gpt-6-luna` agent reviewed the source, requested modern-Python skill, normalized data, provenance hash, README and all four desktop/mobile captures. It did not edit the app.

Initial finding: returning from menu details hid the focused view without restoring keyboard focus. Fixed by returning focus to the originating list row (or list region if no row remains); covered by desktop and mobile browser checks.

Final assessment: suitable for a **draft PR** only, with the missing source export prominent. The two supplied CIA notes preserve Bookbinder's 1977 year, Middle East Restaurant's original name and unknown date, source record links, uncertainty and metadata-only caveat. No guessed pins, items or prices appear in production data. The screenshots show the caveat, empty map counts and provenance without obvious clipping or overflow.

Remaining limit: schema validation proves shape and relationships, not whether a future export matches its claimed source. The full NYPL/CIA import, map placement and real transcription rendering cannot be approved from this preview while the source requests return HTTP 403. New data needs an independent data review after import. The synthetic browser-response fixture checks rendering mechanics only and is not evidence of a real historical menu.

The review suggested making it even clearer the starter records came from supplied parent notes rather than a fresh source download. The visible banner was updated to say “two catalog records from supplied research notes.”
