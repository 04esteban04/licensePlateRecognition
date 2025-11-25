-- Create database if not exists
DO $$
BEGIN
   PERFORM 1 FROM pg_database WHERE datname = 'car_plate';
   IF NOT FOUND THEN
      EXECUTE 'CREATE DATABASE car_plate';
   END IF;
END$$;

\connect car_plate

CREATE SCHEMA IF NOT EXISTS lpr;
SET search_path = lpr, public;

DROP TABLE IF EXISTS car_data CASCADE;
CREATE TABLE car_data (
    id BIGSERIAL PRIMARY KEY,
    plate_class TEXT NOT NULL,
    plate_number TEXT NOT NULL,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    body TEXT NOT NULL,
    year INT NOT NULL,
    engine_cc INT NOT NULL,
    fuel TEXT,
    cabin TEXT,
    traction TEXT,
    transmission TEXT,
    alerts TEXT DEFAULT 'None',
    full_plate TEXT GENERATED ALWAYS AS (
        CASE
            WHEN plate_class = 'CL' THEN trim(plate_class || plate_number)
            ELSE trim(plate_number)
        END
    ) STORED,
    CONSTRAINT unique_plate UNIQUE (plate_class, plate_number)
);

CREATE INDEX idx_cardata_full_plate ON car_data(full_plate);

DROP TABLE IF EXISTS inference_records;
CREATE TABLE inference_records (
    id BIGSERIAL PRIMARY KEY,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    detected_plate TEXT NOT NULL,
    car_id BIGINT REFERENCES car_data(id) ON DELETE SET NULL
);

INSERT INTO car_data
(plate_class, plate_number, brand, model, body, year, engine_cc, fuel, cabin, traction, transmission, alerts) VALUES
('CL','410909','MAZDA','BT-50','PICK-UP OPEN BED TRUCK',2018,2198,'DIESEL','DOUBLE','4X4','AUTOMATIC','Has pending fines. Vehicle cannot be transferred until debts are cleared.'),
('CL','322352','TOYOTA','TACOMA','PICK-UP OPEN BED TRUCK',2002,2400,'GASOLINE','SINGLE','4X2','MANUAL',DEFAULT),
('PART','894232','NISSAN','TIIDA','SEDAN 4 DOORS',2012,1798,'GASOLINE','SINGLE','4X2','MANUAL',DEFAULT),
('PART','VLV495','VOLVO','EX 30 CORE SM','SUV 4 DOORS',2024,0,'ELECTRIC','SINGLE','4X2',NULL, DEFAULT),
('PART','888649','NISSAN','XTERRA','SUV 4 DOORS',2005,4000,'GASOLINE','SINGLE','4X2','MANUAL',DEFAULT),
('PART','DBZ025','VOLKSWAGEN','JETTA MK6','SEDAN 4 DOORS',2016,2000,'GASOLINE','SINGLE','4X2','MANUAL','Stolen vehicle'),
('PART','837258','HONDA','ACCORD','SEDAN 4 DOORS',2009,3500,'GASOLINE','SINGLE','4X2','AUTOMATIC',DEFAULT),
('PART','RYM022','BMW','330 CI','CONVERTIBLE',2002,3000,'GASOLINE','SINGLE','4X2','AUTOMATIC',DEFAULT),
('PART','478280','VOLKSWAGEN','JETTA GL','SEDAN 4 DOORS',1994,2000,'GASOLINE','SINGLE','4X2','MANUAL','Has an active investigation under case reference CR-2019-527.'),
('PART','836908','HYUNDAI','ACCENT','SEDAN 4 DOORS',1997,1500,'GASOLINE','SINGLE','4X2','MANUAL',DEFAULT),
('PART','472799','MITSUBISHI','MONTERO GL','SUV 4 DOORS',2002,2835,'DIESEL','SINGLE','4X4','MANUAL',DEFAULT),
('PART','BZG765','TOYOTA','4RUNNER','SUV 4 DOORS',2007,4000,'GASOLINE','N/A','4X4','MANUAL',DEFAULT),
('PART','QRM014','TOYOTA','LAND CRUISER PRADO T','SUV 4 DOORS',2010,4000,'GASOLINE','SINGLE','4X4','AUTOMATIC',DEFAULT),
('CL','246489','DONGFENG','RICH','PICK-UP OPEN BED TRUCK',2010,3153,'DIESEL','DOUBLE','4X4','MANUAL',DEFAULT),
('PART','BTC886','UAZ','HUNTER 315148','SUV 4 DOORS',2019,2235,'DIESEL','SINGLE','4X4','MANUAL','Expired insurance policy detected. Renewal required before vehicle can circulate legally.'),
('PART','541853','MITSUBISHI','LANCER GLX','SEDAN 4 DOORS',2004,1584,'GASOLINE','SINGLE','4X2','AUTOMATIC',DEFAULT),
('PART','825583','MITSUBISHI','MONTERO','SUV 4 DOORS',2001,3500,'GASOLINE','SINGLE','4X4','AUTOMATIC',DEFAULT),
('CL','368563','VOLKSWAGEN','AMAROK','PICK-UP OPEN BED TRUCK',2014,2000,'DIESEL','DOUBLE','4X4','MANUAL',DEFAULT),
('CL','331976','NISSAN','FRONTIER','PICK-UP OPEN BED TRUCK',2005,2500,'GASOLINE','EXTENDED','4X2','AUTOMATIC','Irregular import documentation under review by Customs Authority'),
('PART', '874562', 'RENAULT', 'KOLEOS', 'SUV 4 DOORS', 2011, 2500, 'GASOLINE', 'SINGLE', '4X4', 'AUTOMATIC', DEFAULT);

INSERT INTO inference_records (detected_plate, car_id)
SELECT 'CL410909', id FROM carData WHERE full_plate = 'CL410909';

INSERT INTO inference_records (detected_plate, car_id)
VALUES ('DBZ025', (SELECT id FROM carData WHERE full_plate = 'DBZ025')),
       ('ABC123', NULL);
