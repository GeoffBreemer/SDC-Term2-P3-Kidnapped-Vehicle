/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <iostream>
#include <map>

#include "particle_filter.h"

using namespace std;

// Code employs a hack that uses the id property to assign measurements to landmarks
#define ID_NOT_ASSIGNED -1

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles
	num_particles = 50;

	// Create Normal distributions around the initial position and heading estimate
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Initialise the weights vector to 1
	weights.resize((unsigned long) num_particles, 1.0);

	// Initialise the particles vector
	default_random_engine gen;
	particles.resize((unsigned long) num_particles);

	for (int i=0;i<particles.size();i++) {
		struct Particle p;

		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles[i] = p;
	}

	// Done initialising
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Initialise the random number generator
  default_random_engine gen;

  // Set measurement yaw
  double m_yaw = yaw_rate * delta_t;

  // Add measurement and gaussian noise to each particle
  for (int i=0;i<particles.size();i++) {
    Particle *p = &particles[i];
		double yaw = p->theta;

    // Add measurement
		if (fabs(yaw_rate) > 0.001) {
			// Update x, y and heading
			p->x += velocity / yaw_rate * (sin(yaw + m_yaw) - sin(yaw));
			p->y += velocity / yaw_rate * (cos(yaw) - cos(yaw + m_yaw));
			p->theta += m_yaw;
		} else {
			// Update x and y only for close-to-zero yaw rates
			p->x += velocity * delta_t * cos(yaw);
			p->y += velocity * delta_t * sin(yaw);
		}

		// Add gaussian noise
		normal_distribution<double> dist_x(p->x, std_pos[0]);
		normal_distribution<double> dist_y(p->y, std_pos[1]);
		normal_distribution<double> dist_theta(p->theta, std_pos[2]);
		p->x = dist_x(gen);
		p->y = dist_y(gen);
		p->theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // Loop through all predicted landmarks and assign each to one observation
	for(int i=0;i<predicted.size();i++) {
    LandmarkObs *prediction = &predicted[i];

    double closest_dist = numeric_limits<double>::max();   // set initial closest distance to a large number
    LandmarkObs* prv_closest = NULL;  // no previous assignment available at the start

    // Loop through all observations
    for(int j=0;j<observations.size();j++) {
      LandmarkObs* observation = &observations[j];

			// Skip observations already assigned
      if (observation->id != ID_NOT_ASSIGNED)
				continue;

      // Calculate the distance
			double d = dist(observation->x, observation->y, prediction->x, prediction->y);

			// Update to the current observation if the distance is smaller
			if (d < closest_dist) {
				// Allow previously assigned one to be picked again
				if (prv_closest != NULL)
					prv_closest->id = -1;

				// Assign to the predicted landmark
				observation->id = prediction->id;
				prv_closest = observation;
				closest_dist = d;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

  // Loop through all particles
  for (int i=0;i<particles.size();i++) {
    Particle *p = &particles[i];

    // Convert observations to map space
		vector<LandmarkObs> observations_map(observations);

    for(int j=0;j<observations_map.size();j++) {
      LandmarkObs *obs = &observations_map[j];
			double vx = obs->x * cos(p->theta) - obs->y * sin(p->theta) + p->x;
			double vy = obs->x * sin(p->theta) + obs->y * cos(p->theta) + p->y;
			obs->x = vx;
			obs->y = vy;
			obs->id = ID_NOT_ASSIGNED;
		}

		// Predict landmarks within sensor range of this particle
		vector<LandmarkObs> predicted;
    for (int j=0;j<map_landmarks.landmark_list.size();j++) {
      Map::single_landmark_s landmark;
      landmark = map_landmarks.landmark_list[j];
			if (dist(p->x, p->y, landmark.x_f, landmark.y_f) < sensor_range) {
				predicted.push_back({landmark.id_i, landmark.x_f, landmark.y_f});
			}
		}

		// Find nearest neighbor
		dataAssociation(predicted, observations_map);

		// Lookup predicted landmark
		map<int, LandmarkObs> predictedMap;
		for (auto prediction : predicted) {
			predictedMap.insert({prediction.id, prediction});
		}

		// Calculate MVN weight
		double weight_prod = 1;
    for(int j=0;j<observations_map.size();j++) {
      LandmarkObs *measurement = &observations_map[j];
			LandmarkObs predicted_measurement = predictedMap[measurement->id];

      // Calculate the weight
			double part1 = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
			double part2 = pow(measurement->x - predicted_measurement.x, 2) / pow(std_landmark[0], 2);
			double part3 = pow(measurement->y - predicted_measurement.y, 2) / pow(std_landmark[1], 2);
			double weight = part1 * exp(-0.5 * (part2 + part3));

      // Avoid too small values
			if (weight < .0001)
				weight = .0001;

      // Add to the total sum
			weight_prod *= weight;
		}

    // Assign weight
		p->weight = weight_prod;
	}
}

void ParticleFilter::resample() {
	// Intialise resampling wheel
	double beta = 0.0;
  double max_weight = 0.0;
  default_random_engine genix;

	uniform_int_distribution<int> uni_disc(0, num_particles - 1);
	int ix = uni_disc(genix);

	vector<Particle> resampled_particles;
	resampled_particles.reserve((unsigned long) num_particles);

	// Find maximum weight and create particle weights vectors
	weights.clear();
	weights.reserve((unsigned long) num_particles);
  for(int i=0;i<particles.size();i++) {
    Particle *p = &particles[i];
		if (p->weight > max_weight)
			max_weight = p->weight;
		weights.push_back(p->weight);
	}

  // Resample
  default_random_engine gen;
	uniform_real_distribution<double> uni_cont(0, 2.0 * max_weight);
  for (int i=0;i<num_particles;i++) {
		beta += uni_cont(gen);
		while (weights[ix] < beta) {
			beta -= weights[ix];
			ix = (ix + 1) % num_particles;
		}
		resampled_particles.push_back(particles[ix]);
	}

	// Update particles
	particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
