import spidev
import RPi.GPIO as GPIO
import time
import csv
import numpy as np


CS_PINS = [8, 7, 5, 6]  # GPIO numbers for CS1, CS2, CS3, CS4
DEVID       = 0x00
POWER_CTL   = 0x2D
DATA_FORMAT = 0x31
DATAX0      = 0x32
READ_BIT    = 0x80
MULTI_BIT   = 0x40
MAX_PLAUSIBLE_G = 6.0  # reject any reading above this as a glitched sample

LOG_FILE = "accel_log.csv"

GPIO.setmode(GPIO.BCM)
for pin in CS_PINS:
	GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)  # idle high to match pull-up

spi = spidev.SpiDev()
spi.open(0, 0)
spi.no_cs = True  # stop the kernel from also auto toggling CE0 (GPIO8), drive CS manually
spi.max_speed_hz = 1000000
spi.mode = 0b11  # ADXL345 uses SPI mode 3

def select(pin):
	GPIO.output(pin, GPIO.LOW)

def deselect(pin):
	GPIO.output(pin, GPIO.HIGH)

def read_reg(pin, reg, length=1):
	select(pin)
	cmd = reg | READ_BIT | (MULTI_BIT if length > 1 else 0)
	result = spi.xfer2([cmd] + [0x00] * length)
	deselect(pin)
	return result[1:]

def write_reg(pin, reg, value):
	select(pin)
	spi.xfer2([reg, value])
	deselect(pin)

# check each sensor, DEVID should read back as 0xE5
for i, pin in enumerate(CS_PINS):
	devid = read_reg(pin, DEVID)[0]
	print(f"Sensor{i+1} (CS pin {pin}): DEVID = 0x{devid:02X}", "OK" if devid == 0xE5 else "MISMATCHED")

# write/readback diagnostic: proves two-way SPI communication independent of DEVID
for i, pin in enumerate(CS_PINS):
	write_reg(pin, DATA_FORMAT, 0x2B)  # arbitrary distinctive test value
	readback = read_reg(pin, DATA_FORMAT)[0]
	print(f"Sensor{i+1} (CS pin {pin}): wrote 0x2B, read back 0x{readback:02X}", "OK" if readback == 0x2B else "MISMATCHED")

# config and start measuring
for pin in CS_PINS:
	write_reg(pin, DATA_FORMAT, 0x08)  # full res, +/-2g
	write_reg(pin, POWER_CTL, 0x08)    # measurement mode

def read_accel(pin):
	raw = read_reg(pin, DATAX0, 6)
	def to_signed16(lo, hi):
		val = (hi << 8) | lo
		return val - 65536 if val > 32767 else val
	x = to_signed16(raw[0], raw[1]) * 0.0039  # 4mg/LSB in full res mode
	y = to_signed16(raw[2], raw[3]) * 0.0039
	z = to_signed16(raw[4], raw[5]) * 0.0039
	if max(abs(x), abs(y), abs(z)) > MAX_PLAUSIBLE_G:
		return None  # reject glitched sample (e.g. stuck-byte SPI dropout)
	return x, y, z

def sensor_rotation(phi_deg):
    phi = np.radians(phi_deg)
    return np.array([
        [-np.sin(phi),  np.cos(phi), 0],
        [ np.cos(phi),  np.sin(phi), 0],
        [ 0,             0,          1]
    ])

# One rotation matrix per sensor, built once at startup
SENSOR_PHI = {1: 90, 2: 0, 3: 270, 4: 180}
R = {i: sensor_rotation(phi) for i, phi in SENSOR_PHI.items()}

def to_global(sensor_id, a_local):
    """a_local = (ax, ay, az) from that sensor's raw reading"""
    return R[sensor_id] @ np.array(a_local)

def format_timestamp(elapsed_sec):
    """Format elapsed seconds as MM:SS.mmm"""
    minutes = int(elapsed_sec // 60)
    seconds = int(elapsed_sec % 60)
    millis = int((elapsed_sec - int(elapsed_sec)) * 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"

# Rows get appended here each loop iteration, written out to CSV once at the end
log_rows = []
start_time = time.monotonic()

try:
	while True:
		elapsed = time.monotonic() - start_time
		ts = format_timestamp(elapsed)
		for i, pin in enumerate(CS_PINS):
			result = read_accel(pin)
			if result is None:
				print(f"Sensor {i+1}: [glitch rejected]", end="  ")
				continue
			x, y, z = result
			gx, gy, gz = to_global(i + 1, (x, y, z))
			print(f"Sensor {i+1}: local x={x:+.3f} y={y:+.3f} z={z:+.3f}  |  global X={gx:+.3f} Y={gy:+.3f} Z={gz:+.3f}", end="  ")
			log_rows.append([ts, elapsed, i + 1, x, y, z, gx, gy, gz])
		print(f" [{ts}]")
		time.sleep(0.02)  # 50Hz loop, 1/50Hz = 0.02s
except KeyboardInterrupt:
	pass
finally:
	GPIO.cleanup()
	spi.close()
	with open(LOG_FILE, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["timestamp_mm_ss_ms", "t_sec", "sensor_id", "local_x", "local_y", "local_z", "global_X", "global_Y", "global_Z"])
		writer.writerows(log_rows)
	print(f"\nLogged {len(log_rows)} rows to {LOG_FILE}")
