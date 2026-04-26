#define enA 9
#define in1 6
#define in2 7
String message;
int speed = 0;

void setup() {
  pinMode(enA, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(13, OUTPUT);
  // Set initial rotation direction
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  Serial.begin(115200);
}

void loop() {
  int potValue = analogRead(A0); // Read potentiometer value
  int pwmOutput = map(potValue, 0, 1023, 0 , 255); // Map the potentiometer value from 0 to 255
  if(speed > 160)
  {
    digitalWrite(13, HIGH);
  }
  else
  {
    digitalWrite(13, LOW);
  }
  
  if(pwmOutput > 110)
  {
    analogWrite(enA, speed); // Send PWM signal to L298N Enable pin
  }
  else
  {
    analogWrite(enA, 0); // Send PWM signal to L298N Enable pin
  }
  
  if(Serial.available() > 0)
  {
    Serial.setTimeout(10);
    message = Serial.readString();
    Serial.println(message);
    speed = message.toInt();
  }
}