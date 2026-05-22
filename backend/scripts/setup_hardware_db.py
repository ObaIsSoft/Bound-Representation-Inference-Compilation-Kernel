"""
hardware_db Supabase migration + seed script.

Creates the hardware_db table and seeds 6 popular MCU specs.
Run once from the backend/ directory:

    python3 scripts/setup_hardware_db.py

If the table doesn't exist yet, the script prints the DDL SQL to run
in the Supabase SQL editor, then exits. Re-run after creating the table.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
from supabase import create_client

sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_SERVICE_KEY'])

# ─── DDL ─────────────────────────────────────────────────────────────────────

DDL = """
-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New query)
CREATE TABLE IF NOT EXISTS hardware_db (
  id         UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  mcu_key    TEXT        UNIQUE NOT NULL,
  mcu_name   TEXT        NOT NULL,
  family     TEXT,
  spec       JSONB       NOT NULL,
  source     TEXT        DEFAULT 'seed',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS hardware_db_mcu_key_idx ON hardware_db (mcu_key);
COMMENT ON TABLE hardware_db IS 'MCU hardware specs for CodegenAgent pin allocation';
"""

# ─── Seed data ────────────────────────────────────────────────────────────────

HARDWARE_SPECS = [
    {
        "mcu_key":  "esp32",
        "mcu_name": "ESP32-WROOM-32",
        "family":   "esp32",
        "source":   "seed",
        "spec": {
            "clock_mhz": 240, "flash_mb": 4, "sram_kb": 520,
            "adc_channels": 18, "dac_channels": 2,
            "pins": {
                "GPIO0":  {"restrictions": ["strapping_pin"]},
                "GPIO2":  {"restrictions": ["strapping_pin"]},
                "GPIO5":  {"restrictions": ["strapping_pin"]},
                "GPIO6":  {"restrictions": ["internal_flash"]},
                "GPIO7":  {"restrictions": ["internal_flash"]},
                "GPIO8":  {"restrictions": ["internal_flash"]},
                "GPIO9":  {"restrictions": ["internal_flash"]},
                "GPIO10": {"restrictions": ["internal_flash"]},
                "GPIO11": {"restrictions": ["internal_flash"]},
                "GPIO12": {"restrictions": ["strapping_pin"]},
                "GPIO15": {"restrictions": ["strapping_pin"]},
                "GPIO34": {"restrictions": ["input_only"], "electrical": {"input_only": True}},
                "GPIO35": {"restrictions": ["input_only"], "electrical": {"input_only": True}},
                "GPIO36": {"restrictions": ["input_only"], "electrical": {"input_only": True}},
                "GPIO39": {"restrictions": ["input_only"], "electrical": {"input_only": True}},
            },
            "peripherals": {
                "i2c": [
                    {"name": "I2C0", "scl": "GPIO22", "sda": "GPIO21", "max_freq_khz": 400},
                    {"name": "I2C1", "scl": "GPIO26", "sda": "GPIO25", "max_freq_khz": 400},
                ],
                "spi": [
                    {"name": "SPI2", "sclk": "GPIO18", "miso": "GPIO19", "mosi": "GPIO23",
                     "cs_pool": ["GPIO5", "GPIO4", "GPIO13", "GPIO14", "GPIO17", "GPIO27", "GPIO33"]},
                    {"name": "SPI3", "sclk": "GPIO14", "miso": "GPIO13", "mosi": "GPIO27",
                     "cs_pool": ["GPIO15", "GPIO32", "GPIO33", "GPIO4"]},
                ],
                "uart": [
                    {"name": "UART0", "tx": "GPIO1",  "rx": "GPIO3"},
                    {"name": "UART1", "tx": "GPIO17", "rx": "GPIO16"},
                    {"name": "UART2", "tx": "GPIO32", "rx": "GPIO33"},
                ],
                "pwm": {
                    "name": "LEDC", "channels": 16,
                    "pins": ["GPIO4","GPIO13","GPIO14","GPIO16","GPIO17","GPIO18",
                             "GPIO19","GPIO21","GPIO22","GPIO23","GPIO25","GPIO26",
                             "GPIO27","GPIO32","GPIO33"],
                },
                "can": [{"name": "TWAI", "tx": "GPIO21", "rx": "GPIO22"}],
                "gpio_cs_pool": ["GPIO4","GPIO13","GPIO14","GPIO17","GPIO27","GPIO33"],
            },
        },
    },
    {
        "mcu_key":  "esp32_s3",
        "mcu_name": "ESP32-S3",
        "family":   "esp32",
        "source":   "seed",
        "spec": {
            "clock_mhz": 240, "flash_mb": 8, "sram_kb": 512,
            "adc_channels": 20, "dac_channels": 0,
            "pins": {
                "GPIO0":  {"restrictions": ["strapping_pin"]},
                "GPIO3":  {"restrictions": ["strapping_pin"]},
                "GPIO45": {"restrictions": ["strapping_pin"]},
                "GPIO46": {"restrictions": ["strapping_pin"]},
                "GPIO26": {"restrictions": ["internal_flash"]},
                "GPIO27": {"restrictions": ["internal_flash"]},
                "GPIO28": {"restrictions": ["internal_flash"]},
                "GPIO29": {"restrictions": ["internal_flash"]},
                "GPIO30": {"restrictions": ["internal_flash"]},
                "GPIO31": {"restrictions": ["internal_flash"]},
                "GPIO32": {"restrictions": ["internal_flash"]},
                "GPIO33": {"restrictions": ["internal_flash"]},
            },
            "peripherals": {
                "i2c": [
                    {"name": "I2C0", "scl": "GPIO9",  "sda": "GPIO8",  "max_freq_khz": 400},
                    {"name": "I2C1", "scl": "GPIO18", "sda": "GPIO17", "max_freq_khz": 400},
                ],
                "spi": [
                    {"name": "SPI2", "sclk": "GPIO12", "miso": "GPIO13", "mosi": "GPIO11",
                     "cs_pool": ["GPIO10", "GPIO14", "GPIO15", "GPIO16"]},
                ],
                "uart": [
                    {"name": "UART0", "tx": "GPIO43", "rx": "GPIO44"},
                    {"name": "UART1", "tx": "GPIO17", "rx": "GPIO18"},
                ],
                "pwm": {
                    "name": "LEDC", "channels": 8,
                    "pins": ["GPIO1","GPIO2","GPIO4","GPIO5","GPIO6","GPIO7",
                             "GPIO8","GPIO9","GPIO10","GPIO11","GPIO12","GPIO13",
                             "GPIO14","GPIO15","GPIO16","GPIO17","GPIO18","GPIO19",
                             "GPIO20","GPIO21"],
                },
                "gpio_cs_pool": ["GPIO10","GPIO14","GPIO15","GPIO16","GPIO21"],
            },
        },
    },
    {
        "mcu_key":  "stm32f405",
        "mcu_name": "STM32F405RG",
        "family":   "stm32f4",
        "source":   "seed",
        "spec": {
            "clock_mhz": 168, "flash_kb": 1024, "sram_kb": 192,
            "adc_channels": 16, "dac_channels": 2,
            "pins": {
                "PA13": {"restrictions": ["debug_pin"], "note": "SWDIO"},
                "PA14": {"restrictions": ["debug_pin"], "note": "SWCLK"},
                "PB3":  {"restrictions": ["debug_pin"], "note": "SWO (optional)"},
            },
            "peripherals": {
                "i2c": [
                    {"name": "I2C1", "scl": "PB6",  "sda": "PB7",  "max_freq_khz": 400},
                    {"name": "I2C2", "scl": "PB10", "sda": "PB11", "max_freq_khz": 400},
                    {"name": "I2C3", "scl": "PA8",  "sda": "PC9",  "max_freq_khz": 400},
                ],
                "spi": [
                    {"name": "SPI1", "sclk": "PA5",  "miso": "PA6",  "mosi": "PA7",
                     "cs_pool": ["PA4","PB0","PB1","PC4","PC5"]},
                    {"name": "SPI2", "sclk": "PB13", "miso": "PB14", "mosi": "PB15",
                     "cs_pool": ["PB12","PC6","PC7","PC8"]},
                    {"name": "SPI3", "sclk": "PC10", "miso": "PC11", "mosi": "PC12",
                     "cs_pool": ["PA15","PB4","PB5"]},
                ],
                "uart": [
                    {"name": "USART1", "tx": "PA9",  "rx": "PA10"},
                    {"name": "USART2", "tx": "PA2",  "rx": "PA3"},
                    {"name": "USART3", "tx": "PB10", "rx": "PB11"},
                    {"name": "UART4",  "tx": "PC10", "rx": "PC11"},
                ],
                "can": [
                    {"name": "CAN1", "tx": "PA12", "rx": "PA11"},
                    {"name": "CAN2", "tx": "PB13", "rx": "PB12"},
                ],
                "pwm": {
                    "name": "TIM_PWM",
                    "pins": ["PA0","PA1","PA2","PA3","PA6","PA7","PA8","PA9","PA10","PA11",
                             "PA15","PB0","PB1","PB4","PB5","PB6","PB7","PB8","PB9",
                             "PB14","PB15","PC6","PC7","PC8","PC9"],
                },
                "gpio_cs_pool": ["PA4","PB0","PB1","PB12","PC4","PC5","PC6","PC7"],
            },
        },
    },
    {
        "mcu_key":  "rp2040",
        "mcu_name": "RP2040 (Raspberry Pi Pico)",
        "family":   "rp2040",
        "source":   "seed",
        "spec": {
            "clock_mhz": 133, "flash_mb": 2, "sram_kb": 264,
            "adc_channels": 4, "pio_state_machines": 8,
            "pins": {
                "GPIO23": {"restrictions": ["smps_control"]},
                "GPIO24": {"restrictions": ["vbus_detect"]},
                "GPIO25": {"restrictions": ["onboard_led"], "note": "Pico built-in LED"},
            },
            "peripherals": {
                "i2c": [
                    {"name": "I2C0", "scl": "GPIO5",  "sda": "GPIO4",  "max_freq_khz": 400},
                    {"name": "I2C1", "scl": "GPIO7",  "sda": "GPIO6",  "max_freq_khz": 400},
                ],
                "spi": [
                    {"name": "SPI0", "sclk": "GPIO18", "miso": "GPIO16", "mosi": "GPIO19",
                     "cs_pool": ["GPIO17","GPIO2","GPIO3","GPIO14","GPIO15"]},
                    {"name": "SPI1", "sclk": "GPIO10", "miso": "GPIO12", "mosi": "GPIO11",
                     "cs_pool": ["GPIO13","GPIO20","GPIO21","GPIO22"]},
                ],
                "uart": [
                    {"name": "UART0", "tx": "GPIO0", "rx": "GPIO1"},
                    {"name": "UART1", "tx": "GPIO8", "rx": "GPIO9"},
                ],
                "pwm": {
                    "name": "PWM",
                    "pins": [f"GPIO{i}" for i in range(0, 23)],  # GPIO0-GPIO22
                },
                "gpio_cs_pool": ["GPIO2","GPIO3","GPIO14","GPIO15","GPIO20","GPIO21","GPIO22"],
            },
        },
    },
    {
        "mcu_key":  "nrf52840",
        "mcu_name": "nRF52840",
        "family":   "nrf52",
        "source":   "seed",
        "spec": {
            "clock_mhz": 64, "flash_kb": 1024, "sram_kb": 256,
            "adc_channels": 8, "ble": True, "ieee_802_15_4": True,
            "pins": {},
            "peripherals": {
                "i2c": [
                    {"name": "TWI0", "scl": "P0_08", "sda": "P0_09", "max_freq_khz": 400},
                    {"name": "TWI1", "scl": "P0_11", "sda": "P0_12", "max_freq_khz": 400},
                ],
                "spi": [
                    {"name": "SPI0", "sclk": "P0_14", "miso": "P0_15", "mosi": "P0_16",
                     "cs_pool": ["P0_13","P0_17","P0_18","P0_19"]},
                    {"name": "SPI1", "sclk": "P0_29", "miso": "P0_30", "mosi": "P0_31",
                     "cs_pool": ["P0_28","P1_01","P1_02","P1_03"]},
                ],
                "uart": [
                    {"name": "UART0", "tx": "P0_05", "rx": "P0_06"},
                    {"name": "UART1", "tx": "P0_07", "rx": "P0_08"},
                ],
                "pwm": {
                    "name": "PWM",
                    "pins": [f"P0_{i:02d}" for i in range(2, 32) if i not in (6, 10, 18, 24, 25, 26, 27)] +
                            [f"P1_{i:02d}" for i in range(0, 16)],
                },
                "gpio_cs_pool": ["P0_13","P0_17","P0_18","P0_19","P0_28","P1_01","P1_02"],
            },
        },
    },
    {
        "mcu_key":  "atmega2560",
        "mcu_name": "ATmega2560 (Arduino Mega)",
        "family":   "avr",
        "source":   "seed",
        "spec": {
            "clock_mhz": 16, "flash_kb": 256, "sram_kb": 8,
            "adc_channels": 16,
            "pins": {},
            "peripherals": {
                "i2c": [
                    {"name": "Wire", "scl": "21", "sda": "20", "max_freq_khz": 400},
                ],
                "spi": [
                    {"name": "SPI", "sclk": "52", "miso": "50", "mosi": "51",
                     "cs_pool": ["53","10","11","12","13"]},
                ],
                "uart": [
                    {"name": "Serial",  "tx": "1",  "rx": "0"},
                    {"name": "Serial1", "tx": "18", "rx": "19"},
                    {"name": "Serial2", "tx": "16", "rx": "17"},
                    {"name": "Serial3", "tx": "14", "rx": "15"},
                ],
                "pwm": {
                    "name": "PWM",
                    "pins": ["2","3","4","5","6","7","8","9","10","11","12","13",
                             "44","45","46"],
                },
                "gpio_cs_pool": ["53","10","22","23","24","25","26"],
            },
        },
    },
    {
        "mcu_key":  "teensy41",
        "mcu_name": "Teensy 4.1",
        "family":   "imxrt1062",
        "source":   "seed",
        "spec": {
            "clock_mhz": 600, "flash_mb": 8, "sram_kb": 1024,
            "adc_channels": 18, "ethernet": True,
            "pins": {},
            "peripherals": {
                "i2c": [
                    {"name": "Wire",  "scl": "19", "sda": "18", "max_freq_khz": 1000},
                    {"name": "Wire1", "scl": "16", "sda": "17", "max_freq_khz": 1000},
                    {"name": "Wire2", "scl": "24", "sda": "25", "max_freq_khz": 1000},
                ],
                "spi": [
                    {"name": "SPI",  "sclk": "13", "miso": "12", "mosi": "11",
                     "cs_pool": ["10","9","8","7","6"]},
                    {"name": "SPI1", "sclk": "27", "miso": "1",  "mosi": "26",
                     "cs_pool": ["0","38","39","40"]},
                ],
                "uart": [
                    {"name": "Serial1", "tx": "1",  "rx": "0"},
                    {"name": "Serial2", "tx": "8",  "rx": "7"},
                    {"name": "Serial3", "tx": "14", "rx": "15"},
                    {"name": "Serial4", "tx": "17", "rx": "16"},
                    {"name": "Serial5", "tx": "20", "rx": "21"},
                    {"name": "Serial6", "tx": "24", "rx": "25"},
                    {"name": "Serial7", "tx": "28", "rx": "29"},
                    {"name": "Serial8", "tx": "34", "rx": "35"},
                ],
                "can": [
                    {"name": "CAN1", "tx": "22", "rx": "23"},
                    {"name": "CAN2", "tx": "30", "rx": "31"},
                ],
                "pwm": {
                    "name": "FlexPWM",
                    "pins": [str(i) for i in range(0, 40)],
                },
                "gpio_cs_pool": ["10","9","8","7","6","38","39","40"],
            },
        },
    },
]

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n=== BRICK OS — hardware_db migration ===\n")

    # 1. Check if table exists
    try:
        sb.table('hardware_db').select('id').limit(1).execute()
        print("✓ Table hardware_db exists.")
    except Exception as e:
        if 'PGRST205' in str(e) or 'schema cache' in str(e):
            print("✗ Table hardware_db does not exist.\n")
            print("Run this SQL in the Supabase SQL Editor:")
            print("  Dashboard → SQL Editor → New query → paste → Run\n")
            print("─" * 60)
            print(DDL)
            print("─" * 60)
            print("\nThen re-run this script to seed the data.")
            sys.exit(1)
        else:
            raise

    # 2. Upsert all hardware specs
    inserted = 0
    updated  = 0
    for hw in HARDWARE_SPECS:
        try:
            # Check if already seeded
            existing = sb.table('hardware_db') \
                .select('id,mcu_key') \
                .eq('mcu_key', hw['mcu_key']) \
                .execute()
            if existing.data:
                sb.table('hardware_db') \
                    .update({'spec': hw['spec'], 'mcu_name': hw['mcu_name'],
                             'family': hw['family']}) \
                    .eq('mcu_key', hw['mcu_key']) \
                    .execute()
                print(f"  ↺ Updated:  {hw['mcu_name']}  ({hw['mcu_key']})")
                updated += 1
            else:
                sb.table('hardware_db').insert(hw).execute()
                print(f"  ✓ Inserted: {hw['mcu_name']}  ({hw['mcu_key']})")
                inserted += 1
        except Exception as e:
            print(f"  ✗ Failed ({hw['mcu_key']}): {e}")

    print(f"\nDone — {inserted} inserted, {updated} updated.")

    # 3. Verify
    result = sb.table('hardware_db').select('mcu_key,mcu_name').order('mcu_key').execute()
    print(f"\nhardware_db rows ({len(result.data)} total):")
    for row in result.data:
        print(f"  {row['mcu_key']:20s} {row['mcu_name']}")
    print()


if __name__ == '__main__':
    main()
