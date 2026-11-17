# Encrypted backup of the learned model

These files are AES-256 encrypted (`openssl enc -aes-256-cbc -pbkdf2 -iter 200000`). Without the password they
are unreadable. The password is not in this repository: Yazan keeps it.

- `adapter_gen2.pt.enc.part0`..`part4`: the learned version (generation 2), trained on consented voices, split
  into pieces under GitHub's 100 MB limit. Never publish it decrypted.
- `pronlex.json.enc`: the full pronunciation dictionary with per-person counts.

Restore on the laptop (it asks for the password):

    cat backup/adapter_gen2.pt.enc.part* | openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -out runs/learn/adapter.pt
    openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -in backup/pronlex.json.enc -out runs/learn/pronlex.json

If someone withdraws consent, delete these files and push again; a later backup is made without their voice.
