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

### Commands

```bash
cd test_cases
form-amdl.d 2 famdl.l9bi.d.202c amdl.l9bi.d.202c
redistrb.cy.d redistrb.sun.prxt3.in > ttt.red.sun.out
adipls.c.d adipls.sun.in > ttt.adi.sun.out
```

(or ` cd workingdir && make`)

### Input file format (adipls.sun.in)

Note that according to section 7.2.1 of `notes/adiab.prg.v0_3.pdf`, only lines ending with `@` are read.

Some quick notes:
- To get all the eigenfunctions in a range of frequencies, one should additionally set `itrsig/=1`, and set the variables `sig1`, `sig2`, and `iscan`.

## Output format

See section 8.1 of `notes/adiab.prg.v0_3.pdf` for the format used for the output.

To read the eigenfunction output into Python, use the function `read_modes` from `read_output.py`

# Plot scripts

The following scripts run the calculations for the indicated cases and plot k-omega diagrams

- plot_truncplane.py: truncated radial extent, plane-parallel geometry, Cowling approximation
- plot_trunc.py: truncated radial extent, spherical geometry, Cowling approximation
- plot_cowling.py: full radial extent, Cowling approximation
- plot.py: full non-Cowling calculation
