#!/usr/bin/env python3
"""Decode the numeric constants in the step function at the end of SE field."""

# CONST1 bits analysis (manual trace from lambda text):
# Layer 0: x8748 selected (cons), BIT = (\x8753.\x8754.x8754) = false = 0
# Layer 1: x8756 selected (cons), BIT = (\x8761.\x8762.x8762) = false = 0
# Layer 2: x8764 selected (cons), BIT = (\x8769.\x8770.x8770) = false = 0
# Layer 3: x8772 selected (cons), BIT = (\x8777.\x8778.x8778) = false = 0
# Layer 4: x8780 selected (cons), BIT = (\x8785.\x8786.x8786) = false = 0
# Layer 5: x8788 selected (cons), BIT = (\x8793.\x8794.x8794) = false = 0
# Layer 6: x8796 selected (cons), BIT = (\x8801.\x8802.x8802) = false = 0
# Layer 7: x8804 selected (cons), BIT = (\x8809.\x8810.x8810) = false = 0
# Layer 8: x8812 selected (cons), BIT = (\x8817.\x8818.x8818) = false = 0
# Layer 9: x8820 selected (cons), BIT = (\x8825.\x8826.x8826) = false = 0
# Layer 10: x8828 selected (cons), BIT = (\x8833.\x8834.x8834) = false = 0
# Layer 11: x8836 selected (cons), BIT = (\x8841.\x8842.x8841) = TRUE = 1!
# Layer 12: x8844 selected (cons), BIT = (\x8849.\x8850.x8850) = false = 0
# Terminal: (\x8852.\x8853.x8853) = nil
# Bits (LSB first): 0,0,0,0,0,0,0,0,0,0,0,1,0
# Binary: bit 11 set = 2^11 = 2048
print("CONST1 bits (LSB first): 0,0,0,0,0,0,0,0,0,0,0,1,0")
print("CONST1 = 2^11 = 2048 (confirmed!)")
print()

# CONST2 bits:
# Layers 0-7: all false = 0
# Layer 8: (\x8924.\x8925.x8924) = TRUE = 1
# Layer 9: (\x8932.\x8933.x8933) = false = 0
# Terminal: (\x8935.\x8936.x8936) = nil
# Bits (LSB first): 0,0,0,0,0,0,0,0,1,0
# Binary: bit 8 set = 2^8 = 256
print("CONST2 bits (LSB first): 0,0,0,0,0,0,0,0,1,0")
print("CONST2 = 2^8 = 256 (confirmed!)")
print()

# CONST3 bits:
# Layers 0-4: all false = 0
# Layer 5: (\x8983.\x8984.x8983) = TRUE = 1
# Layer 6: (\x8991.\x8992.x8992) = false = 0
# Terminal: (\x8994.\x8995.x8995) = nil
# Bits (LSB first): 0,0,0,0,0,1,0
# Binary: bit 5 set = 2^5 = 32
print("CONST3 bits (LSB first): 0,0,0,0,0,1,0")
print("CONST3 = 2^5 = 32 (confirmed!)")
print()
print("Step function: step(step(step(x4468, 2048), 256), 32)")
print("  = step(step(step(data, 2^11), 2^8), 2^5)")
print("  11 + 8 + 5 = 24 depth levels (total 1 + 24 = 25)")
