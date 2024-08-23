#!/bin/bash
set -e

echo ""
echo "#### form-amdl.d ####"
form-amdl.d 2 famdl.l9bi.d.202c amdl.l9bi.d.202c

echo ""
echo "#### redistrb.cy.d ####"
redistrb.cy.d redistrb.sun.prxt3.in > ttt.red.sun.out

echo ""
echo "#### adipls.c.d ####"
adipls.c.d adipls.sun.in > ttt.adi.sun.out

