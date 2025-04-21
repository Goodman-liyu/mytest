from sre_constants import IN


INSTRUCTIONS = {
    "Rot13": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Examples:
    * Example 1:
        - **Ciphertext**:
            Anzr: Rzvyl Wbuafba; Qngr bs Ovegu: Whyl 15, 1990; Cnffcbeg Ahzore: L987654321
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            Pnfr Ahzore: 2024-CF-001234; Pnfr Glcr: Gursg/Oernx-Va; Svyvat Qngr: Bpgbore 19, 2024
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            gnk_vqragvsvre: GKA-2023-NOP456, gnk_erpbeqf: {lrne: 2023, fgnghf: Cebprffrq, ershaq_vffhrq: 620.00}
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "Atbash": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Examples:
    * Example 1:
        - **Ciphertext**:
            Mznv: Vnrob Qlsmhlm; Wzgv lu Yrigs: Qfob 15, 1990; Kzhhklig Mfnyvi: B987654321
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            Xzhv Mfnyvi: 2024-KH-001234; Xzhv Gbkv: Gsvug/Yivzp-Rm; Urormt Wzgv: Lxglyvi 19, 2024
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            gzc_rwvmgrurvi: GCM-2023-ZYX456, gzc_ivxliwh: {bvzi: 2023, hgzgfh: Kilxvhhvw, ivufmw_rhhfvw: 620.00}
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "Polybius": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Examples:
    * Example 1:
        - **Ciphertext**:
            32 11 31 15 :   15 31 23 26 51   24 33 22 32 41 33 32 ;   14 11 42 15   33 16   12 23 36 42 22 :   24 43 26 51   1 5 ,   1 9 9 0 ;   34 11 41 41 34 33 36 42   32 43 31 12 15 36 :   51 9 8 7 6 5 4 3 2 1
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            13 11 41 15   32 43 31 12 15 36 :   2 0 2 4 - 34 41 - 0 0 1 2 3 4 ;   13 11 41 15   42 51 34 15 :   42 22 15 16 42 / 12 36 15 11 25 - 23 32 ;   16 23 26 23 32 21   14 11 42 15 :   33 13 42 33 12 15 36   1 9 ,   2 0 2 4
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            42 11 46 _ 23 14 15 32 42 23 16 23 15 36 :   42 46 32 - 2 0 2 3 - 11 12 13 4 5 6 ,   42 11 46 _ 36 15 13 33 36 14 41 :   { 51 15 11 36 :   2 0 2 3 ,   41 42 11 42 43 41 :   34 36 33 13 15 41 41 15 14 ,   36 15 16 43 32 14 _ 23 41 41 43 15 14 :   6 2 0 . 0 0 }
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "Vigenere": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Examples:
    * Example 1:
        - **Ciphertext**:
            Ncxe: Eotla Jqsnuzn; Dcee zf Miteh: Jwwy 15, 1990; Pcdsrzrv Nwxbgc: J987654321
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            Ccde Yuomet: 2024-PU-001234; Naup Vjpg: Vsehe/Dcecv-Ky; Qintni Dcee: Oeeodpr 19, 2024
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            tci_koepeihtet: VIN-2023-CMC456, tci_tpcqcdu: {jecc: 2023, dtceuu: Rcoepsupd, rgqupo_kdswpd: 620.00}
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "Reverse": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Examples:
    * Example 1:
        - **Ciphertext**:
            123456789Y :rebmuN tropssaP ;0991 ,51 yluJ :htriB fo etaD ;nosnhoJ ylimE :emaN
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            4202 ,91 rebotcO :etaD gniliF ;nI-kaerB/tfehT :epyT esaC ;432100-SP-4202 :rebmuN esaC
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            }00.026 :deussi_dnufer ,dessecorP :sutats ,3202 :raey{ :sdrocer_xat ,654CBA-3202-NXT :reifitnedi_xat
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "SwapPairs": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Examples:
    * Example 1:
        - **Ciphertext**:
            aNem :mEli yoJnhos;nD ta efoB riht :uJyl1 ,51 99;0P sapsro tuNbmre :9Y78563412
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            aCesN mueb:r2 20-4SP0-1032;4C sa eyTep :hTfe/trBae-knI ;iFilgnD ta:eO tcbore1 ,92 204
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            at_xdineitifre :XT-N0232A-CB54,6t xar_cerosd :y{ae:r2 20,3s atut:sP orecssde ,erufdni_sseu:d6 020.}0
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "ParityShift": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Examples:
    * Example 1:
        - **Ciphertext**:
            Ozld: Dlhmx Kniorno; Ezud ng Chsui: Ktmx 15, 1990; Qzrrqnsu Otlcds: X987654321
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            Bzrd Otlcds: 2024-QR-001234; Bzrd Uxqd: Uidgu/Csdzj-Ho; Ghmhof Ezud: Nbuncds 19, 2024
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            uzy_hedouhghds: UYO-2023-ZCB456, uzy_sdbnser: {xdzs: 2023, ruzutr: Qsnbdrrde, sdgtoe_hrrtde: 620.00}
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "DualAvgCode": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Examples:
    * Example 1:
        - **Ciphertext**:
            MOaalndf: DFlnhjkmxz IKnpgimortnpmo; CEaasudf npeg AChjqssugi: IKtvkmxz 15, 1990; OQaartrtoqnpqssu MOtvlnacdfqs: XZ987654321
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            BDaartdf MOtvlnacdfqs: 2024-OQRT-001234; BDaartdf SUxzoqdf: SUgidfegsu/ACqsdfaajl-HJmo; EGhjkmhjmofh CEaasudf: NPbdsunpacdfqs 19, 2024
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            suaawy_hjcedfmosuhjeghjdfqs: SUWYMO-2023-AAACBD456, suaawy_qsdfbdnpqscert: {xzdfaaqs: 2023, rtsuaasutvrt: OQqsnpbddfrtrtdfce, qsdfegtvmoce_hjrtrttvdfce: 620.00}
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "WordShift": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Examples:
    * Example 1:
        - **Ciphertext**:
            e:Nam lyEmi nson;Joh eDat fo th:Bir yJul 15, 0;199 sportPas ber:Num 7654321Y98
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            eCas ber:Num 4-PS-001234;202 eCas e:Typ ft/Break-In;The ingFil e:Dat oberOct 19, 4202
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            _identifier:tax -2023-ABC456,TXN _records:tax ar:{ye 3,202 tus:sta cessed,Pro und_issued:ref .00}620
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
}

INSTRUCTIONS2 = {
    "Rot13": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Algorithm Flow:
    Uses the Caesar cipher with a fixed shift of 13 positions. For each letter in the **Plaintext**, shift it forward by 13 positions in the alphabet to produce the **Ciphertext**.
## Examples:
    * Example 1:
        - **Ciphertext**:
            Anzr: Rzvyl Wbuafba; Qngr bs Ovegu: Whyl 15, 1990; Cnffcbeg Ahzore: L987654321
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            Pnfr Ahzore: 2024-CF-001234; Pnfr Glcr: Gursg/Oernx-Va; Svyvat Qngr: Bpgbore 19, 2024
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            gnk_vqragvsvre: GKA-2023-NOP456, gnk_erpbeqf: {lrne: 2023, fgnghf: Cebprffrq, ershaq_vffhrq: 620.00}
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "Atbash": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Algorithm Flow:
    Uses the Atbash cipher. Each letter in the **Plaintext** is replaced with its Reverse counterpart in the alphabet.
## Examples:
    * Example 1:
        - **Ciphertext**:
            Mznv: Vnrob Qlsmhlm; Wzgv lu Yrigs: Qfob 15, 1990; Kzhhklig Mfnyvi: B987654321
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            Xzhv Mfnyvi: 2024-KH-001234; Xzhv Gbkv: Gsvug/Yivzp-Rm; Urormt Wzgv: Lxglyvi 19, 2024
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            gzc_rwvmgrurvi: GCM-2023-ZYX456, gzc_ivxliwh: {bvzi: 2023, hgzgfh: Kilxvhhvw, ivufmw_rhhfvw: 620.00}
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "Polybius": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Algorithm Flow:
    Uses the Polybius cipher. Each letter in the **Plaintext** is mapped to a pair of coordinates in the Polybius square, forming the **Ciphertext**.
## Examples:
    * Example 1:
        - **Ciphertext**:
            32 11 31 15 :   15 31 23 26 51   24 33 22 32 41 33 32 ;   14 11 42 15   33 16   12 23 36 42 22 :   24 43 26 51   1 5 ,   1 9 9 0 ;   34 11 41 41 34 33 36 42   32 43 31 12 15 36 :   51 9 8 7 6 5 4 3 2 1
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            13 11 41 15   32 43 31 12 15 36 :   2 0 2 4 - 34 41 - 0 0 1 2 3 4 ;   13 11 41 15   42 51 34 15 :   42 22 15 16 42 / 12 36 15 11 25 - 23 32 ;   16 23 26 23 32 21   14 11 42 15 :   33 13 42 33 12 15 36   1 9 ,   2 0 2 4
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            42 11 46 _ 23 14 15 32 42 23 16 23 15 36 :   42 46 32 - 2 0 2 3 - 11 12 13 4 5 6 ,   42 11 46 _ 36 15 13 33 36 14 41 :   { 51 15 11 36 :   2 0 2 3 ,   41 42 11 42 43 41 :   34 36 33 13 15 41 41 15 14 ,   36 15 16 43 32 14 _ 23 41 41 43 15 14 :   6 2 0 . 0 0 }
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "Vigenere": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Algorithm Flow:
    Uses the Vigenère cipher. Each letter in the **Plaintext** is shifted by the corresponding letter in the **Key** to produce the **Ciphertext**.
## Examples:
    * Example 1:
        - **Ciphertext**:
            Ncxe: Eotla Jqsnuzn; Dcee zf Miteh: Jwwy 15, 1990; Pcdsrzrv Nwxbgc: J987654321
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            Ccde Yuomet: 2024-PU-001234; Naup Vjpg: Vsehe/Dcecv-Ky; Qintni Dcee: Oeeodpr 19, 2024
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            tci_koepeihtet: VIN-2023-CMC456, tci_tpcqcdu: {jecc: 2023, dtceuu: Rcoepsupd, rgqupo_kdswpd: 620.00}
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "Reverse": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Algorithm Flow:
    Reverses the **Plaintext** to create the **Ciphertext**.
## Examples:
    * Example 1:
        - **Ciphertext**:
            123456789Y :rebmuN tropssaP ;0991 ,51 yluJ :htriB fo etaD ;nosnhoJ ylimE :emaN
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            4202 ,91 rebotcO :etaD gniliF ;nI-kaerB/tfehT :epyT esaC ;432100-SP-4202 :rebmuN esaC
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            }00.026 :deussi_dnufer ,dessecorP :sutats ,3202 :raey{ :sdrocer_xat ,654CBA-3202-NXT :reifitnedi_xat
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "SwapPairs": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Algorithm Flow:
    For each pair of letters in the **Plaintext**, their positions are swapped to produce the **Ciphertext**. If the number of letters is odd, the last letter remains in its original position.
## Examples:
    * Example 1:
        - **Ciphertext**:
            aNem :mEli yoJnhos;nD ta efoB riht :uJyl1 ,51 99;0P sapsro tuNbmre :9Y78563412
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            aCesN mueb:r2 20-4SP0-1032;4C sa eyTep :hTfe/trBae-knI ;iFilgnD ta:eO tcbore1 ,92 204
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            at_xdineitifre :XT-N0232A-CB54,6t xar_cerosd :y{ae:r2 20,3s atut:sP orecssde ,erufdni_sseu:d6 020.}0
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "ParityShift": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Algorithm Flow:
    For each letter in the **Plaintext**: 
        - If the ASCII value is even, add 1 to it to get the corresponding character in the **Ciphertext**.
        - If the ASCII value is odd, subtract 1 to get the new character in the **Ciphertext**.
## Examples:
    * Example 1:
        - **Ciphertext**:
            Ozld: Dlhmx Kniorno; Ezud ng Chsui: Ktmx 15, 1990; Qzrrqnsu Otlcds: X987654321
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            Bzrd Otlcds: 2024-QR-001234; Bzrd Uxqd: Uidgu/Csdzj-Ho; Ghmhof Ezud: Nbuncds 19, 2024
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            uzy_hedouhghds: UYO-2023-ZCB456, uzy_sdbnser: {xdzs: 2023, ruzutr: Qsnbdrrde, sdgtoe_hrrtde: 620.00}
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "DualAvgCode": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Algorithm Flow:
    This encryption method converts each letter of the **Plaintext** into two letters in the **Ciphertext**, such that the average of their ASCII values equals the ASCII value of the original letter.
## Examples:
    * Example 1:
        - **Ciphertext**:
            MOaalndf: DFlnhjkmxz IKnpgimortnpmo; CEaasudf npeg AChjqssugi: IKtvkmxz 15, 1990; OQaartrtoqnpqssu MOtvlnacdfqs: XZ987654321
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            BDaartdf MOtvlnacdfqs: 2024-OQRT-001234; BDaartdf SUxzoqdf: SUgidfegsu/ACqsdfaajl-HJmo; EGhjkmhjmofh CEaasudf: NPbdsunpacdfqs 19, 2024
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            suaawy_hjcedfmosuhjeghjdfqs: SUWYMO-2023-AAACBD456, suaawy_qsdfbdnpqscert: {xzdfaaqs: 2023, rtsuaasutvrt: OQqsnpbddfrtrtdfce, qsdfegtvmoce_hjrtrttvdfce: 620.00}
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
    "WordShift": """## Role:
    Cryptography Analysis Expert.
## Goals:
    Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
## Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results.
## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Algorithm Flow:
    The algorithm splits the **Plaintext** into words based on spaces. Each word is then individually encrypted using the Caesar cipher, resulting in the **Ciphertext**.
## Examples:
    * Example 1:
        - **Ciphertext**:
            e:Nam lyEmi nson;Joh eDat fo th:Bir yJul 15, 0;199 sportPas ber:Num 7654321Y98
        - **Plaintext**:
            Name: Emily Johnson; Date of Birth: July 15, 1990; Passport Number: Y987654321
    * Example 2:
        - **Ciphertext**:
            eCas ber:Num 4-PS-001234;202 eCas e:Typ ft/Break-In;The ingFil e:Dat oberOct 19, 4202
        - **Plaintext**:
            Case Number: 2024-PS-001234; Case Type: Theft/Break-In; Filing Date: October 19, 2024
    * Example 3:
        - **Ciphertext**:
            _identifier:tax -2023-ABC456,TXN _records:tax ar:{ye 3,202 tus:sta cessed,Pro und_issued:ref .00}620
        - **Plaintext**:
            tax_identifier: TXN-2023-ABC456, tax_records: {year: 2023, status: Processed, refund_issued: 620.00}
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:""",
}
