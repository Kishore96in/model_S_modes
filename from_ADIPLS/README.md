# File listing
adipack.v0_3.tar.gz: Downloaded from <https://phys.au.dk/~jcd/adipack.v0_3/adipack.v0_3.tar.gz> (see <https://ascl.net/1109.002>) on 15 Feb 2024, 8:04PM IST.

# Steps to use

## Compilation

```bash
mkdir -p adipack.v0_3
tar -xzf adipack.v0_3.tar.gz -C ./adipack.v0_3

cd adipack.v0_3/adipls/
./setups -C gfortran
make

cd ../adiajobs
make
```

## Things to be done very time you want to use the code

Make sure you are using bash. Note that things seem to break if the path of the directory the code resides in contains spaces.

```bash
ADIPLS_LOCATION="/home/kishore/Documents/Nishant/code/model_S_eigenfunctions/from_ADIPLS/adipack.v0_3" #edit according to your setup.
export PATH="$PATH":"$ADIPLS_LOCATION"/bin
export aprgdir="$ADIPLS_LOCATION"
```

(or source `init.sh`)

## Running a solar model

```bash
cd test_cases
form-amdl.d 2 famdl.l9bi.d.202c amdl.l9bi.d.202c
redistrb.cy.d redistrb.sun.prxt3.in > ttt.red.sun.out
adipls.c.d adipls.sun.in > ttt.adi.sun.out
```

(or ` cd workingdir && ./run.sh`)

## Output format

See section 8.1 of `notes/adiab.prg.v0_3.pdf` for the format used for the output.

