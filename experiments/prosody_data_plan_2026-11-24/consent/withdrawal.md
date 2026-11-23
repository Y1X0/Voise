# Withdrawal and deletion procedure

1. A request can come in any form (message, call, in person), with no reason needed. Log the date of the request
   in the encrypted key file.
2. Within 7 days, delete for that participant code:
   - raw and segmented recordings (all sessions);
   - extracted features and any per-clip or per-session analysis rows;
   - any synthetic clips generated from their voice;
   - their session-log rows; and their name in the key file, which is replaced by "withdrawn".
3. Models: any model or adapter trained with their recordings is taken out of use the same day, and deleted or
   retrained without them within 30 days (training on study recordings only starts after data collection ends).
4. Recompute any aggregate result that included them; a committed aggregate file is updated in the next commit with
   a note "recomputed after a withdrawal" (no code or name given).
5. Backups: delete from every encrypted backup too, and record which backups were cleaned.
6. Send the participant written confirmation of what was deleted and when.
7. Partial withdrawal (one optional use only, e.g. training): remove only that use and any outputs of it, then
   confirm in writing.
