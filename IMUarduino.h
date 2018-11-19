/*****************************************************************
This code:
* Read the gyroscope and accelerometer
  using the readGryo(), readAccel() functions and 
  the gx, gy, gz, ax, ay and az variables.
*****************************************************************/
// The SFE_LSM9DS1 library requires both Wire and SPI be
// included BEFORE including the 9DS1 library.
#include <Wire.h>
#include <SPI.h>
#include <SparkFunLSM9DS1.h>

// LSM9DS1 Library Init //
LSM9DS1 imu1;
LSM9DS1 imu2;

// Example I2C Setup //
// SDO_XM and SDO_G are both pulled high, so our addresses are:
#define LSM9DS1_M1 0x1E // Would be 0x1C if SDO_M is LOW
#define LSM9DS1_AG1  0x6B // Would be 0x6A if SDO_AG is LOW
#define LSM9DS1_M2 0x1C // Would be 0x1C if SDO_M is LOW
#define LSM9DS1_AG2  0x6A // Would be 0x6A if SDO_AG is LOW
int i = 0;

void setup() 
{
  Serial.begin(250000);
  
  imu1.settings.device.commInterface = IMU_MODE_I2C;
  imu1.settings.device.mAddress = LSM9DS1_M1;
  imu1.settings.device.agAddress = LSM9DS1_AG1;

  imu2.settings.device.commInterface = IMU_MODE_I2C;
  imu2.settings.device.mAddress = LSM9DS1_M2;
  imu2.settings.device.agAddress = LSM9DS1_AG2;

  if (!imu1.begin())
  {
    Serial.println("Failed to communicate with LSM9DS1_1_right.");
    while (1) ;
  }
  if (!imu2.begin())
  {
    Serial.println("Failed to communicate with LSM9DS1_2_left.");
    while (1) ;
  }
}

void loop()
{
  // Update the sensor values whenever new data is available 1
  if ( imu1.gyroAvailable() )   imu1.readGyro();
  if ( imu1.accelAvailable() )  imu1.readAccel();
  // Update the sensor values whenever new data is available 2
  if ( imu2.gyroAvailable() )   imu2.readGyro();
  if ( imu2.accelAvailable() )  imu2.readAccel();
  
    Serial.print(i); Serial.print(",");
    Serial.print(int16_t(imu1.ax)); Serial.print(",");
    Serial.print(int16_t(imu1.ay)); Serial.print(",");
    Serial.print(int16_t(imu1.az)); Serial.print(",");
    Serial.print(int16_t(imu1.gx)); Serial.print(",");
    Serial.print(int16_t(imu1.gy)); Serial.print(",");
    Serial.print(int16_t(imu1.gz)); Serial.print(",");
    Serial.print(int16_t(imu2.ax)); Serial.print(",");
    Serial.print(int16_t(imu2.ay)); Serial.print(",");
    Serial.print(int16_t(imu2.az)); Serial.print(",");
    Serial.print(int16_t(imu2.gx)); Serial.print(",");
    Serial.print(int16_t(imu2.gy)); Serial.print(",");
    Serial.print(int16_t(imu2.gz));
    Serial.print("\n");
  i++;
}
