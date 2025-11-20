#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Download the public MPSD dataset
DLCMD="curl --progress-bar -L -o"
DESTINATION=$1

URLS=(
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An_YbpFOEEq-i0gsdv_rs3pS-YCuxroa4SjldwIg6Afpsk5OvkKW1qik7FV_VUB7d5MpMoCsToCqmAp92MazElVn0F_d_Sx06yqB8R1hckzF4A2CEiZMLd76ikNVdSzJQiUolJ8l.zip?ccb=10-5&oh=00_AYDi7Sef26_TKXvcw-sXMndo9UPa7ZihAxsb-FZylykNFg&oe=67B33207&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An_HR43KZvA3xlrihi_Uwy0S1kGfDkPg_4n1nQn8zrJRJGFoQqzUyeBbz8JXIpT6NMNwaJ57rpMO1CZFspPVTntgn_inIIXvstEFjl6EAd8J9VxmaBcsNvGxYXDHAERKwg.zip?ccb=10-5&oh=00_AYAEHKla9ehFigLJzF-fHca9UJ1tq7nGXQbRlnIGtaNITg&oe=67B3405A&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An8f5aynDuONwsdgQ8fjK63SNyklj2f1FKoL6Vo35wj_S4dE9vnp6dT1Kr3JS7eXrjxS92WG4NJqobL2pPP0XnKnxk4cfriZ5bZZgCSt3jKeypQixBR1qDvZ71v87Q7Z2g.zip?ccb=10-5&oh=00_AYBmWKgCVsDwzwlT1USgrp23Jf4Q2R-7GsFvZ1GvK3MLeA&oe=67B33B35&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An8NneZmJ_hjuskNMEv25epzZ_b6mQW-aJlATjC7p6KtwzOQ3IkVFFwXIobRRgKWLUYw8D43nRnHxcjD9c434smf2KgqZrPcuMec0Mmejc_xEmB899tt_1EvmYq8rqY4tA.zip?ccb=10-5&oh=00_AYAGT3xP88auVjZ8VxyXhsWM_pK_kF9QhTKcFcVr7UmltQ&oe=67B356A6&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An8TgTgFecJ2AZWinsQiSSGc2mULvYCFuwN8ExG415Uu5lSuk2Gw8-ftcOTfRcCmXskzxgvAIC4TQad3iMpT3XQQejOlOV0XazrzIrjhVV0HFLVcirYH0VS5ToeG1ndG8g.zip?ccb=10-5&oh=00_AYDGcHim7gY5tZzNaX-o_yAMJEHu7cGdQUSrYS3cgIJlgw&oe=67B342A4&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An8nbpVa2qxWwflhy6Z4jm_zdHya4_O_dB_fjaQTf-7AM3tZpz6Kwxb9Y5s3t6KX_CG4YxUtn6Wg4OyoGc22O43Thtzm4GWdfx_jVE2Zx2nJkaTn51c5JanGdqfv3XmQhg.zip?ccb=10-5&oh=00_AYAy_Glq6Hy8xo94h6ZfxfekJgjduWQRObhlrtLnfIPOIg&oe=67B3422F&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An-zKIdtRtoB361CWl7fbktbdst8zAvszgbnYJloYkK9OQp2XhTYjr3kJUuQDjptjDvAiR-Juq7ty5a6H_eFCUT9pPRqWs833V9mOvwBvabQkIWvdMFPRfMQiV2oxwNPdA.zip?ccb=10-5&oh=00_AYBqIbAzLFA3P3yjNrqYGRdTMdwLXoBqirm-gNd-5hYQbQ&oe=67B32CBF&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An9z6qlJnGRkaSqZM0KtWeGI9Shmw-paQJl_tOj6WgsWIoweUrohszuFDCXFYjwRqRfx1RnEBQrSQ7_IP5A6FI4nPEMxpoO_2H6Kyguccn4TM_r3uzZ7vI1zvAxM3BBQxQ.zip?ccb=10-5&oh=00_AYAZx-ILO93CUGth95rGgppjzMRqIx7VHOWvfyEDDK7yZw&oe=67B33152&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An_bMd_9QtE6ESdH51XUHAZzuwDQSgOmYLdFWSMhGEX62AlISSoe588--boiK187zKsJiMrDgL1lYsfQG7dKlLG8nHT7X5hSEypBmuzlMavJDw3Yzn923hJwINeNOuu1Lg.zip?ccb=10-5&oh=00_AYBIcT3W8txemPv_gPVjyHf9-zePFHXI-Q5RJabRgf3o6g&oe=67B3434D&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An87OsgbLx1KPfZzH5jtKbK-CQVwpaUNDtQdPGEi5hteVcVu5GpBEVYv6lDpf8Xof1nH782HJc6Eq5JLOt_9RtOYt1pE4mD6nLpPrGXtQZE92ixlhrpBLLjwlLFYBidjfg.zip?ccb=10-5&oh=00_AYBBxkrD9AhseQ4HValwZ4os_NqIlw31VXK4WX86b6ClWg&oe=67B35452&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An9PSHwIOpAEgCGYvTPxWqme7Y_4HESPDA6gFN5rRW1aO1ptY_GCfsKB3z5CvdLTnADxwNnU-eyqdk2POcE7pe677jhKYhQio4z7Df9Hgl7aCE2W5LWWbASzclJQhkIh-w.zip?ccb=10-5&oh=00_AYCuNRNPdJDjFuVSpsegqTfAXcGN8WW-tGJM-pXk57jeJg&oe=67B32AF9&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An-DEZAEuCJ5jYES0zfV-pv587dBOWu2Z0xUoQcB3_oHjtwKruwaU7Xa57xkyMNIyg3Hxw1cJCG4gBlsTTQOQ_WBtKgl-VxDP1H8NPUDiNFdsrjc3SrDnO0j81k9a9Ak9A.zip?ccb=10-5&oh=00_AYCymjVgO2HPxoIQ9AEeYk2UnU1S3b0qu8auzjozQHvK5g&oe=67B34291&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An_NJRYjkXlnBDptGen4KOdJEWFnrv5evNufx5JkcIo4fBBfTLIlZQuReM9Ongjp8UnDfXcc9yN1qu3ccFgKx09Pc8VzRxSs77NtmqmPVP79bni9JXRwyc-g27qfUrPDhg.zip?ccb=10-5&oh=00_AYB5Rn_EzIkjhb5ild3X9yy5msyAyOFbNtJvZfi09s46yQ&oe=67B36075&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An8EmMi5r-n_AdDUkudogHS-dF5tEp94IVcsRq1VIJ7YC76NE9CwuY1s8bS2tYgUfx-NEJA1glHVpJCfeUGz286vIYhh3KU5ajdpH1D3yrhI0plA-4KXY6Qx_1HJQjfErg.zip?ccb=10-5&oh=00_AYAW0YfGWtytQXn67AnnKhHajUDSHxYvQ_kWPzkgM4GR-Q&oe=67B33631&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An9Z9KiE_joAu5lTP0GQ2iYPdjip4jhXQW06usP9iat_8OtH65eQzz8DOHXdYZsT6tgFUh_bLXUledmecEWC4lBJI25ntsWeHMf0sKst6qhw-_CSzjn2C2osCi6aCL_XMg.zip?ccb=10-5&oh=00_AYBDfOSVFdkPAY6_Og4CZfceycS3LH8UXdu2fe3RN4HXqQ&oe=67B348BC&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An_8A6uBpoK60mPc2hoHnR--jSDa7kfiMVhP3oUuP6XCbS8iaVz07qm6fAvuFYpmGwaBVjXlYzwOgDDBSjoSYd7ovkUot0vVYmVby3n_AgRm4G4tnAWZbDCRH97b-Um1Sg.zip?ccb=10-5&oh=00_AYCRpL7Nulg0K0vKISjVXIMd7uet0GTwS0_pFZG3U2F5ew&oe=67B32E04&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An-f-Omiq9zbgv5KoFC-k2A7SkE0WKISDTUwaV36VRyOaYhkH94dqwf2GSpjEXsmE2zMbjX3Tif5NJsWl3IcrNGui6kFzwMqZP_8LH1ChWIDgFo9W5i2wi_m8Ssv9v5D2A.zip?ccb=10-5&oh=00_AYCN2ntNBDTf9pfk9RR3Vj6rKGssREQ4VYiPG01LEeL6VA&oe=67B34A04&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An_dYDkJhKg5xO5VsMO2LLx2hROi2BSSRg5qVxofmkzZiY8yzx2ozkQrTAHWoeH61fvlGcvWHnXTaZIlYvhvNrnI_6FBP7xHyHR5oX4PJdHTlNK0bKyG4TZlr987uC1tyw.zip?ccb=10-5&oh=00_AYBlcJYy5f5DgN80UyGdKmEsLa7aIO09TK1zam_5GdyikA&oe=67B34A4A&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An-gdsEZGIKTWXktf_8lEyjb-3eBxYFikg1ItQH-2DVdgeRYSl0YStSN5c8B0KVNwW-8_RHNzhB4-aqmPZ8lrW1-n0H0LcRVFMFLRxgpBk4zqVun1YcnLms_U199v1hJ_g.zip?ccb=10-5&oh=00_AYCxshQ5EwSvI-6cB1DFqThXhg7JwshSFgEnKOP9_pHjRQ&oe=67B33BBF&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An8J2lu0p-zprNwwWSOkKHOodiMwD1dM72EUDogd84laMaTaL4cqBenmiTmHGO8yhPqfN-dsjC-eokUz0cuW9sgly0K6HpYRO3qNlhTlExo6ovUF-MolC6dXFzsIJt8_WA.zip?ccb=10-5&oh=00_AYCixkzUrV7ufVtIp4jR8viHsGL_Dhh_7mdaL7YcacVWLA&oe=67B35604&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An_-eapSSSigmRDBycBHrMBsN6LB2pRqgxurYgpAaoHZXx-n52_QjkJ-Z7ygt-Zn8Ek-EpnWbM1tMYSa-nsVGsmFqrL8dgnmz0iCaUSxJs5f04Sx_C0gJC0ZeNwEQLLEjA.zip?ccb=10-5&oh=00_AYC5C7k9KrA8cgQzZj5TsHQXiGK3IDBzgiQXK4_4r6Q6oA&oe=67B32AF7&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An-FPUv69q0Em3Y4Hhbl9e__EWslRgHCFBAmGvvkeJ-lKg5tPOLxtSKnRwJMiPZdtQPJVurq2gJ-L3gkrb4CeKDefi_bFmKK3NQ-TCAonnl2OXFvDTd_6gRUOOXUKETvaQ.zip?ccb=10-5&oh=00_AYBD3CPSlJbYYS_3st30X_hbSSdfVOgGOEF4kzzr_sp95A&oe=67B359D6&_nc_sid=6de079"
  "https://scontent-iad3-1.xx.fbcdn.net/m1/v/t6/An_zGVbA1qIFWZIUj-yA3Vh76AW41pEn2KH2aIQFMwp9HzVd20qIdvlOeL5cLvDxmx4QeUa-W4Z0xVXkic5nuq_aKLF1zVLNXRH3Esg_RxSiMGxTXIc53hSi2_sbogUWRw.zip?ccb=10-5&oh=00_AYDVWnS0VQvXBsXHWYa9BUhhQp1Pl0ubnGkD8HzM9DXB3Q&oe=67B3425C&_nc_sid=6de079"
  "https://scontent-iad3-2.xx.fbcdn.net/m1/v/t6/An_OA25H1KBJ62HzkrTZ4srXneUiV5SFpmNFvrSHLgn_TK4TGdyA13g4XTh-XQ3lTbCVsCQyAnnhNzP7g4T5rrAJ27vrR-lgxvQDiMOdulC4zZaxKIKxlzc.md?_nc_gid=hmtuSNNnY8Ncm9vcDKfCAg&_nc_oc=AdmSq2xmQ1-EN1bU_VnIkq8O6n-WYLaF4XpGpzkrEPdsHAcRjpYSLl0T1WoP5ZYF9e6icBrhhZSlfudiE7raaA5z&ccb=10-5&oh=00_AfSYXN5g-NyVyaYYVsXe4Ol26vo9mNFof8tNhBzSR9I6dA&oe=689E9B4C&_nc_sid=6de079"
  "https://scontent-iad3-2.xx.fbcdn.net/m1/v/t6/An_dVt3rUNN8wJdzvvNcwfgQh-ebUW0RkRjNZ6ZSgbi96we3LySO_i7h3OtpFzKeyF9kNOyE3utbLt2NbsG-7x4eHVi53XmSKzZxbxFBtslxquShe2HuLFNOV4tB.zip?_nc_gid=hmtuSNNnY8Ncm9vcDKfCAg&_nc_oc=AdlwMhCTrHiNJS2fVrtqyDgFmd8XnRPNDvIgsPNYFDEMtECDJG7N8pjFR2jc_8QQpLiV8sNh05KBYDSjPQrd2Eo6&ccb=10-5&oh=00_AfSsm76b8D_EFom0d9Fc3bhjk8Hwjexj_OF1wjjClRoVLA&oe=689E96A7&_nc_sid=6de079"
  "https://scontent-iad3-2.xx.fbcdn.net/m1/v/t6/An84DdKV175yfWDUytmStxkGTz6ngMGh5Hl8aIFJImwp6XW4_Hvjwfpbzr0qrhmyllGNBAxqez2-9iuMypYLSxDrhRi781eEXx6nCuGDKe4ml5oWhzHOZspWCZdfxr08Z21c.npy?_nc_gid=hmtuSNNnY8Ncm9vcDKfCAg&_nc_oc=AdlE0m5gID3UIXkqCAEq1Ygyijyl06AoMVNDf93dZpInzpCa2-Rtpo2MsfvKMsqgppFWzYjm3lgabeGiIgkrq0vr&ccb=10-5&oh=00_AfRdHSehzSePzcN6f1g_F99lqqR699jjvjLBEx3wgFdg7Q&oe=689E9BEA&_nc_sid=6de079"
  "https://scontent-iad3-2.xx.fbcdn.net/m1/v/t6/An9ruFSo1yDOKXxj1WxfVc8jjyQ17umUADOf2BOKu0r04OCOJdfpSpUtk99BkPoywXT05JD2fCjutvepif7j88s0NQv52xZzcBPto-Zi5XVdi0RzP-z5vPR7Yq00Yo16.npy?_nc_gid=hmtuSNNnY8Ncm9vcDKfCAg&_nc_oc=AdkmiXODwIMT2QBgc746_hKOOEzfO1chw9x04-6A0hJsuW-KEe3oStqTpSYe9tOhV-JWf60yuoLgjyGwCuYpFe86&ccb=10-5&oh=00_AfRHIgWSqRVt3TceyCwT-GuNbxalTkaIAF0_8IDoap9XRQ&oe=689EA55F&_nc_sid=6de079"
)

# Check if destination parameter was provided
if [ -z "$DESTINATION" ]; then
  echo "No destination directory specified. Usage: $0 <destination_directory>"
  exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DESTINATION"
if [ $? -ne 0 ]; then
  echo "Failed to create destination directory: $DESTINATION"
  exit 1
fi

for URL in "${URLS[@]}"; do
  FILE="$DESTINATION/$(basename "$URL" | cut -d '?' -f 1)"
  $DLCMD "$FILE" "$URL"
  if [ $? -eq 0 ]; then
    echo "Download of $(basename "$FILE") completed successfully"
  else
    echo "Download of $(basename "$FILE") failed or was incomplete."
  fi
done

echo "All files have been downloaded to $DESTINATION."
