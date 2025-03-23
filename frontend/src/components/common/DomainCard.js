import React from 'react';
import {
  Box,
  Heading,
  Text,
  Badge,
  useColorModeValue,
  VStack,
  HStack,
} from '@chakra-ui/react';

const DomainCard = ({ domain, onClick }) => {
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  
  // Generate a consistent color for each domain
  const getColorScheme = (key) => {
    const options = ['blue', 'green', 'purple', 'red', 'orange', 'teal', 'cyan', 'pink'];
    const index = key.charCodeAt(0) % options.length;
    return options[index];
  };
  
  const colorScheme = getColorScheme(domain.key);
  
  return (
    <Box
      bg={cardBg}
      borderWidth="1px"
      borderRadius="lg"
      borderColor={borderColor}
      overflow="hidden"
      boxShadow="md"
      p={4}
      cursor="pointer"
      onClick={() => onClick(domain)}
      transition="all 0.2s"
      _hover={{ transform: 'translateY(-2px)', boxShadow: 'lg' }}
    >
      <VStack align="start" spacing={3}>
        <Heading size="md" color={`${colorScheme}.500`}>
          {domain.name}
        </Heading>
        
        <Text fontSize="sm" color="gray.600">
          {domain.description}
        </Text>
        
        <HStack flexWrap="wrap">
          {domain.examples?.map((example, index) => (
            <Badge key={index} colorScheme={colorScheme} mr={1} mb={1}>
              {example}
            </Badge>
          ))}
        </HStack>
        
        <Text fontSize="xs" color="gray.500">
          {Object.keys(domain.subdomains || {}).length} subdomains
        </Text>
      </VStack>
    </Box>
  );
};

export default DomainCard;