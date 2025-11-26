import React, { useState } from 'react';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Heading,
  Select,
  Text,
  useToast,
  Badge,
  VStack,
  HStack,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Card,
  CardBody,
  CardHeader,
  useColorModeValue,
  SimpleGrid,
  Divider,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Spinner,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
} from '@chakra-ui/react';
import { useQuery } from '@tanstack/react-query';
import { getDatasets } from '../services/api';

function QualityMetrics() {
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeTab, setActiveTab] = useState(0);

  const toast = useToast();
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  // Fetch datasets
  const { data: datasets, isLoading: isDatasetsLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: getDatasets
  });

  // Full analysis
  const analyzeFullQuality = async () => {
    const response = await fetch(
      `http://localhost:8000/api/quality/analyze/full?dataset_id=${selectedDataset}`,
      { method: 'POST' }
    );
    if (!response.ok) throw new Error('Failed to analyze quality');
    return response.json();
  };

  // Anonymize PII
  const anonymizePII = async () => {
    const response = await fetch(
      `http://localhost:8000/api/quality/anonymize/pii?dataset_id=${selectedDataset}`,
      { method: 'POST' }
    );
    if (!response.ok) throw new Error('Failed to anonymize PII');
    return response.json();
  };

  // Handle full analysis
  const handleFullAnalysis = async () => {
    if (!selectedDataset) {
      toast({
        title: 'No dataset selected',
        description: 'Please select a dataset to analyze',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setIsAnalyzing(true);
    try {
      const results = await analyzeFullQuality();
      setAnalysisResults(results);
      toast({
        title: 'Analysis complete',
        description: 'Quality metrics have been calculated',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Analysis failed',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle anonymization
  const handleAnonymization = async () => {
    if (!selectedDataset) {
      toast({
        title: 'No dataset selected',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    try {
      const result = await anonymizePII();
      toast({
        title: 'PII Anonymized',
        description: result.message,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Anonymization failed',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  return (
    <Box>
      <Heading size="md" mb={6}>Advanced Quality Metrics (Phase 6)</Heading>

      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} mb={6}>
        <CardHeader>
          <Heading size="sm">Dataset Selection</Heading>
        </CardHeader>
        <CardBody>
          <HStack spacing={4}>
            <FormControl maxW="400px">
              <FormLabel>Select Dataset</FormLabel>
              <Select
                value={selectedDataset || ''}
                onChange={(e) => setSelectedDataset(e.target.value ? parseInt(e.target.value) : null)}
                placeholder="Select a dataset"
                isDisabled={isDatasetsLoading}
              >
                {datasets?.datasets?.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name} ({dataset.example_count} examples)
                  </option>
                ))}
              </Select>
            </FormControl>

            <Button
              colorScheme="blue"
              onClick={handleFullAnalysis}
              isDisabled={!selectedDataset}
              isLoading={isAnalyzing}
              mt={8}
            >
              Run Full Analysis
            </Button>
          </HStack>
        </CardBody>
      </Card>

      {isAnalyzing && (
        <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} mb={6}>
          <CardBody>
            <VStack spacing={4}>
              <Spinner size="xl" color="blue.500" />
              <Text>Analyzing dataset quality...</Text>
              <Text fontSize="sm" color="gray.500">
                This may take a few minutes for large datasets
              </Text>
            </VStack>
          </CardBody>
        </Card>
      )}

      {analysisResults && (
        <>
          <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} mb={6}>
            <CardHeader>
              <Heading size="sm">Quality Summary</Heading>
            </CardHeader>
            <CardBody>
              <SimpleGrid columns={{ base: 1, md: 4 }} spacing={6}>
                <Stat>
                  <StatLabel>Total Examples</StatLabel>
                  <StatNumber>{analysisResults.summary.total_examples}</StatNumber>
                </Stat>

                <Stat>
                  <StatLabel>Toxic Content</StatLabel>
                  <StatNumber color={analysisResults.summary.toxic_percentage > 10 ? 'red.500' : 'green.500'}>
                    {analysisResults.summary.toxic_count}
                  </StatNumber>
                  <StatHelpText>
                    {analysisResults.summary.toxic_percentage}% of dataset
                  </StatHelpText>
                </Stat>

                <Stat>
                  <StatLabel>PII Detected</StatLabel>
                  <StatNumber color={analysisResults.summary.pii_percentage > 5 ? 'orange.500' : 'green.500'}>
                    {analysisResults.summary.pii_count}
                  </StatNumber>
                  <StatHelpText>
                    {analysisResults.summary.pii_percentage}% of dataset
                  </StatHelpText>
                </Stat>

                <Stat>
                  <StatLabel>Duplicate Groups</StatLabel>
                  <StatNumber color={analysisResults.summary.duplicate_percentage > 20 ? 'yellow.500' : 'green.500'}>
                    {analysisResults.summary.duplicate_groups}
                  </StatNumber>
                  <StatHelpText>
                    {analysisResults.summary.duplicate_percentage}% duplicates
                  </StatHelpText>
                </Stat>
              </SimpleGrid>

              {/* Quality Alerts */}
              <VStack spacing={3} mt={6} align="stretch">
                {analysisResults.summary.toxic_percentage > 10 && (
                  <Alert status="error">
                    <AlertIcon />
                    <AlertTitle>High Toxicity Detected!</AlertTitle>
                    <AlertDescription>
                      {analysisResults.summary.toxic_percentage}% of your dataset contains toxic content.
                    </AlertDescription>
                  </Alert>
                )}

                {analysisResults.summary.pii_percentage > 5 && (
                  <Alert status="warning">
                    <AlertIcon />
                    <AlertTitle>PII Found!</AlertTitle>
                    <AlertDescription>
                      {analysisResults.summary.pii_count} examples contain personally identifiable information.
                      <Button ml={4} size="sm" colorScheme="orange" onClick={handleAnonymization}>
                        Anonymize PII
                      </Button>
                    </AlertDescription>
                  </Alert>
                )}

                {analysisResults.summary.duplicate_percentage > 20 && (
                  <Alert status="info">
                    <AlertIcon />
                    <AlertTitle>High Duplication Rate</AlertTitle>
                    <AlertDescription>
                      {analysisResults.summary.duplicate_percentage}% of your dataset contains duplicates.
                    </AlertDescription>
                  </Alert>
                )}

                {analysisResults.summary.toxic_percentage <= 10 && 
                 analysisResults.summary.pii_percentage <= 5 && 
                 analysisResults.summary.duplicate_percentage <= 20 && (
                  <Alert status="success">
                    <AlertIcon />
                    <AlertTitle>Excellent Quality!</AlertTitle>
                    <AlertDescription>
                      Your dataset passes all quality checks.
                    </AlertDescription>
                  </Alert>
                )}
              </VStack>
            </CardBody>
          </Card>

          <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
            <CardBody>
              <Tabs index={activeTab} onChange={setActiveTab}>
                <TabList>
                  <Tab>Toxicity Analysis</Tab>
                  <Tab>PII Detection</Tab>
                  <Tab>Duplicate Detection</Tab>
                </TabList>

                <TabPanels>
                  {/* Toxicity Tab */}
                  <TabPanel>
                    <VStack spacing={4} align="stretch">
                      <Text fontWeight="bold">Toxicity Detection Results</Text>
                      
                      <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                        <Stat>
                          <StatLabel>Analyzed</StatLabel>
                          <StatNumber>{analysisResults.toxicity.total_analyzed}</StatNumber>
                        </Stat>
                        <Stat>
                          <StatLabel>Toxic Examples</StatLabel>
                          <StatNumber color="red.500">{analysisResults.toxicity.toxic_count}</StatNumber>
                        </Stat>
                        <Stat>
                          <StatLabel>Toxicity Rate</StatLabel>
                          <StatNumber>{analysisResults.toxicity.toxic_percentage}%</StatNumber>
                        </Stat>
                      </SimpleGrid>

                      <Divider />

                      {analysisResults.toxicity.results && analysisResults.toxicity.results.length > 0 && (
                        <Accordion allowToggle>
                          <AccordionItem>
                            <AccordionButton>
                              <Box flex="1" textAlign="left">
                                View Toxic Examples (First 10)
                              </Box>
                              <AccordionIcon />
                            </AccordionButton>
                            <AccordionPanel>
                              <Table variant="simple" size="sm">
                                <Thead>
                                  <Tr>
                                    <Th>Example ID</Th>
                                    <Th>Status</Th>
                                    <Th>Toxicity</Th>
                                    <Th>Severe</Th>
                                    <Th>Obscene</Th>
                                    <Th>Threat</Th>
                                    <Th>Insult</Th>
                                  </Tr>
                                </Thead>
                                <Tbody>
                                  {analysisResults.toxicity.results.slice(0, 10).filter(r => r.is_toxic).map((result, idx) => (
                                    <Tr key={idx}>
                                      <Td>{result.example_id || 'N/A'}</Td>
                                      <Td>
                                        <Badge colorScheme={result.is_toxic ? 'red' : 'green'}>
                                          {result.is_toxic ? 'Toxic' : 'Clean'}
                                        </Badge>
                                      </Td>
                                      <Td>{(result.scores.toxicity * 100).toFixed(1)}%</Td>
                                      <Td>{(result.scores.severe_toxicity * 100).toFixed(1)}%</Td>
                                      <Td>{(result.scores.obscene * 100).toFixed(1)}%</Td>
                                      <Td>{(result.scores.threat * 100).toFixed(1)}%</Td>
                                      <Td>{(result.scores.insult * 100).toFixed(1)}%</Td>
                                    </Tr>
                                  ))}
                                </Tbody>
                              </Table>
                            </AccordionPanel>
                          </AccordionItem>
                        </Accordion>
                      )}
                    </VStack>
                  </TabPanel>

                  {/* PII Tab */}
                  <TabPanel>
                    <VStack spacing={4} align="stretch">
                      <Text fontWeight="bold">PII Detection Results</Text>
                      
                      <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                        <Stat>
                          <StatLabel>Analyzed</StatLabel>
                          <StatNumber>{analysisResults.pii.total_analyzed}</StatNumber>
                        </Stat>
                        <Stat>
                          <StatLabel>Examples with PII</StatLabel>
                          <StatNumber color="orange.500">{analysisResults.pii.pii_count}</StatNumber>
                        </Stat>
                        <Stat>
                          <StatLabel>PII Rate</StatLabel>
                          <StatNumber>{analysisResults.pii.pii_percentage}%</StatNumber>
                        </Stat>
                      </SimpleGrid>

                      <Divider />

                      {analysisResults.pii.pii_types && Object.keys(analysisResults.pii.pii_types).length > 0 && (
                        <>
                          <Text fontWeight="bold" mt={4}>PII Types Detected:</Text>
                          <Table variant="simple" size="sm">
                            <Thead>
                              <Tr>
                                <Th>Entity Type</Th>
                                <Th isNumeric>Count</Th>
                              </Tr>
                            </Thead>
                            <Tbody>
                              {Object.entries(analysisResults.pii.pii_types).map(([type, count]) => (
                                <Tr key={type}>
                                  <Td>{type}</Td>
                                  <Td isNumeric>{count}</Td>
                                </Tr>
                              ))}
                            </Tbody>
                          </Table>
                        </>
                      )}

                      <Button colorScheme="orange" onClick={handleAnonymization} mt={4}>
                        Anonymize All PII
                      </Button>
                    </VStack>
                  </TabPanel>

                  {/* Duplicates Tab */}
                  <TabPanel>
                    <VStack spacing={4} align="stretch">
                      <Text fontWeight="bold">Duplicate Detection Results</Text>
                      
                      <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                        <Stat>
                          <StatLabel>Analyzed</StatLabel>
                          <StatNumber>{analysisResults.duplicates.total_analyzed}</StatNumber>
                        </Stat>
                        <Stat>
                          <StatLabel>Duplicate Groups</StatLabel>
                          <StatNumber color="yellow.500">{analysisResults.duplicates.duplicate_groups}</StatNumber>
                        </Stat>
                        <Stat>
                          <StatLabel>Total Duplicates</StatLabel>
                          <StatNumber>{analysisResults.duplicates.total_duplicates}</StatNumber>
                        </Stat>
                      </SimpleGrid>

                      <Divider />

                      {analysisResults.duplicates.groups && analysisResults.duplicates.groups.length > 0 && (
                        <Accordion allowToggle>
                          <AccordionItem>
                            <AccordionButton>
                              <Box flex="1" textAlign="left">
                                View Duplicate Groups (First 10)
                              </Box>
                              <AccordionIcon />
                            </AccordionButton>
                            <AccordionPanel>
                              <VStack spacing={3} align="stretch">
                                {analysisResults.duplicates.groups.slice(0, 10).map((group, idx) => (
                                  <Box key={idx} p={3} borderWidth="1px" borderRadius="md">
                                    <Text fontWeight="bold">Group {idx + 1}:</Text>
                                    <Text fontSize="sm" color="gray.500">
                                      {group.group.length} similar examples found
                                    </Text>
                                    <Text fontSize="xs" mt={1}>
                                      Document IDs: {group.group.join(', ')}
                                    </Text>
                                  </Box>
                                ))}
                              </VStack>
                            </AccordionPanel>
                          </AccordionItem>
                        </Accordion>
                      )}
                    </VStack>
                  </TabPanel>
                </TabPanels>
              </Tabs>
            </CardBody>
          </Card>
        </>
      )}
    </Box>
  );
}

export default QualityMetrics;
