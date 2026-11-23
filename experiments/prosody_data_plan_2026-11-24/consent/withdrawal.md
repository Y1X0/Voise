# Withdrawal and deletion procedure

1. A request can come in any form (message, call, in person), with no reason needed. Log the date of the request
   in the encrypted key file.
2. Within 7 days, delete for that participant code:
   - raw and segmented recordings (all sessions);
   - extracted features and any per-clip or per-session analysis rows;
   - any synthetic clips generated from their voice, and any model or adapter trained with their recordings
     (retrain without them if needed);
   - their session-log rows; and their name in the key file, which is replaced by "withdrawn".
3. Recompute any aggregate result that included them; a committed aggregate file is updated in the next commit with
   a note "recomputed after a withdrawal" (no code or name given).
4. Backups: delete from every encrypted backup too, and record which backups were cleaned.
5. Send the participant written confirmation of what was deleted and when.
6. Partial withdrawal (one optional use only, e.g. training): remove only that use and any outputs of it, then
   confirm in writing.
