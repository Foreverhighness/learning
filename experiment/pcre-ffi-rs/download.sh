#!/bin/bash
set -e

NAME="pcre2"
VERSION="10.42"
FULLNAME="${NAME}-${VERSION}"

PCRE2_URL="https://github.com/PCRE2Project/pcre2/releases/download/${FULLNAME}/${FULLNAME}.tar.bz2"
SIG_URL="https://github.com/PCRE2Project/pcre2/releases/download/${FULLNAME}/${FULLNAME}.tar.bz2.sig"

wget ${PCRE2_URL}
wget ${SIG_URL}

# Verify
gpg --fetch-keys https://ftp.exim.org/pub/exim/Public-Key
gpg --verify ${FULLNAME}.tar.bz2.sig ${FULLNAME}.tar.bz2

FOLDER="${PWD}/${NAME}"
mkdir -p ${FOLDER}
tar xjf ${FULLNAME}.tar.bz2 -C ${FOLDER} --strip-components=1

