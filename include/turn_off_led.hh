#ifndef _INCLUDE_TURN_OFF_LED_HH_
#define _INCLUDE_TURN_OFF_LED_HH_

#include <Adafruit_NeoPixel.h>

#define PIN 48
#define NUMPIXELS 1

Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);
#define DELAYVAL 500

void
turn_off_led()
{
    pixels.begin();
    pixels.clear();

    for (int i = 0; i < NUMPIXELS; i++) {
        pixels.setPixelColor(i, pixels.Color(0, 0, 0));
        pixels.show();
        delay(DELAYVAL);
    }
}

#endif  // _INCLUDE_TURN_OFF_LED_HH_
